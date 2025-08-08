import numpy as np
import torch
import os
from datetime import datetime, timedelta
from typing import Tuple, Union, Optional
from transformers import AutoTokenizer
from realtime_codec_agent.utils.llamacpp_utils import LlamaForAlternatingCodeChannels
from dataclasses import dataclass

from .audio_tokenizer import AudioTokenizer
from .utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim


class RealtimeAgentResources:
    def __init__(
        self, 
        llm_model_path: str = "Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-2-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-2-BF16.gguf", 
        codec_model: str = "MagiCodec-50Hz-Base", 
        codec_device: Union[str, torch.device] = None,
    ):
        self.llm = LlamaForAlternatingCodeChannels(
            model_path=llm_model_path,
            n_ctx=131072,
            n_gpu_layers=-1,
            verbose=False,
            flash_attn=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(llm_model_path))
        self.audio_tokenizer = AudioTokenizer(codec_model=codec_model, device=codec_device)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)
    
@dataclass
class RealtimeAgentConfig:
    agent_opening_text: str = None
    agent_voice_enrollment: Tuple[int, np.ndarray] = None
    agent_identity: str = "A"
    user_identity: str = "B"
    temperature: float = 1.0
    top_k: int = 100
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    chunk_size_secs: float = 0.1
    chunk_fade_secs: float = 0.02
    max_context_secs: float = 80.0
    trim_by_secs: float = 20.0
    seed: Optional[int] = None
    header_agent_token: str = "<|agent|>"
    header_agent_voice_token: str = "<|agent_voice|>"
    header_speaker_token: str = "<|speaker|>"
    end_header_token: str = "<|end_header|>"
    start_audio_token: str = "<|audio|>"
    end_audio_token: str = "<|end_audio|>"

    def __post_init__(self):
        if int(self.chunk_size_secs*100) % 2 != 0:
            raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
        if self.chunk_fade_secs > self.chunk_size_secs:
            raise ValueError("Chunk fade length cannot be longer than the chunk size.")
        
class RealtimeAgent:
    def __init__(self, resources: RealtimeAgentResources = None, config: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        self.resources = resources

        if config is None:
            config = RealtimeAgentConfig()
        self.set_config(config)

        self.reset()

    def reset(self):
        self.resources.audio_tokenizer.reset_context()
        c = self.config

        self.resources.llm.init_sampler_for_generate(
            top_k=c.top_k,
            top_p=c.top_p,
            min_p=c.min_p,
            temp=c.temperature,
        )
        self.resources.llm.reset()
        
        voice_enrollment = np.zeros(self.resources.audio_tokenizer.sampling_rate * 3, dtype=np.float32) \
            if c.agent_voice_enrollment is None else c.agent_voice_enrollment
        enrollment_audio_str = self.resources.audio_tokenizer.chunked_tokenize_audio(voice_enrollment, c.chunk_size_secs)
        
        agent_prompt = "".join([
            c.header_agent_token,
            c.header_speaker_token,
            f" {c.agent_identity}",
            c.header_speaker_token,
            f" {c.user_identity}",
            c.header_agent_voice_token,
            enrollment_audio_str,
            c.end_header_token,
        ])
        self.input_ids = self.resources.tokenizer(agent_prompt, return_tensors="pt").input_ids
        self.trim_pos = self.input_ids.shape[-1]
        if c.agent_opening_text:
            agent_prompt += f" {c.agent_identity}: {c.agent_opening_text}"
        agent_prompt += c.start_audio_token
        self.input_ids = self.resources.tokenizer(agent_prompt, return_tensors="pt").input_ids
        self.resources.llm.eval(self.input_ids[0, :-1].tolist())

        self.generated_tokens = 0
        self.generated_audio_tokens = 0
        self.audio_history_ch1, self.audio_history_ch2 = np.zeros((2, 0), dtype=np.float32)
        self.transcript = []
        if c.agent_opening_text:
            self.transcript.append({
                "speaker": c.agent_identity,
                "text": c.agent_opening_text,
                "time_secs": 0.0,
            })
        self.start_time = datetime.now()

    def set_config(self, config: RealtimeAgentConfig):
        self.config = config
        
        self.chunk_size_samples = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.sampling_rate)
        self.chunk_size_frames = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.framerate * 2)

        self.max_context_frames = int(self.config.max_context_secs * self.resources.audio_tokenizer.framerate * 2)
        self.trim_by_frames = int(self.config.trim_by_secs * self.resources.audio_tokenizer.framerate * 2)

        self.crossfade_ramps = create_crossfade_ramps(self.resources.audio_tokenizer.sampling_rate, fade_secs=self.config.chunk_fade_secs)

        self.end_header_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_header_token)
        self.start_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.start_audio_token)
        self.end_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_audio_token)

    def trim_sequences(self) -> None:
        if self.generated_audio_tokens > 0 and (self.generated_audio_tokens * 2) % self.max_context_frames == 0:
            audio_tokens_idx = torch.where(self.input_ids > self.end_header_token_id)[1]
            audio_tokens_idx = audio_tokens_idx[audio_tokens_idx >= self.trim_pos]
            trim_to_pos = audio_tokens_idx[self.trim_by_frames]
            self.input_ids = torch.cat(
                [
                    self.input_ids[..., :self.trim_pos],
                    self.input_ids[..., trim_to_pos:],
                ], 
                dim=1,
            )

    def process_audio_input_ids(self, audio_chunk_input_ids: torch.LongTensor) -> torch.LongTensor:
        out_chunk_input_ids = torch.zeros((1, 0), dtype=audio_chunk_input_ids.dtype)
        for i in range(audio_chunk_input_ids.shape[-1]):
            # trim the sequences to the maximum length
            self.trim_sequences()
            while True:
                # predict next token
                audio_mode = (self.input_ids[0, -2:] > self.end_header_token_id).all()
                last_n = 2 if audio_mode else 1
                next_token = next(self.resources.llm.generate(self.input_ids[0, -last_n:].tolist(), reset=False))
                next_token = torch.LongTensor([[next_token]])
                self.input_ids = torch.cat([self.input_ids, next_token], dim=1)
                self.generated_tokens += 1
                # if next token is an audio token, append the next input (user) audio token
                if next_token[0, 0] > self.end_header_token_id:
                    next_input_token = audio_chunk_input_ids[..., i:i+1]
                    self.input_ids = torch.cat([self.input_ids, next_input_token], dim=1)
                    out_chunk_input_ids = torch.cat([out_chunk_input_ids, next_token], dim=1)
                    self.generated_audio_tokens += 1
                    break # move on to next input token
        return out_chunk_input_ids
    
    def process_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        # Sanity check - input size
        if audio_chunk.shape[-1] != self.chunk_size_samples:
            raise ValueError(f"audio_chunk must have length {self.chunk_size_samples}, but got {audio_chunk.shape[-1]}")

        # Encode audio chunk to input ids
        audio_chunk_str = self.resources.audio_tokenizer.tokenize_audio(audio_chunk)
        audio_chunk_input_ids = self.resources.tokenizer(audio_chunk_str, add_special_tokens=False, return_tensors="pt").input_ids
        
        # Generate next audio chunk (interleaved by frame with the input audio chunk) and also generate any interleaved text
        out_chunk_input_ids = self.process_audio_input_ids(audio_chunk_input_ids)

        # Decode input ids to audio chunk and append to the audio history
        out_chunk_str = self.resources.tokenizer.decode(out_chunk_input_ids[0], skip_special_tokens=False)
        (_, out_chunk), _, preroll_samples = self.resources.audio_tokenizer.detokenize_audio(out_chunk_str, preroll_samples=self.crossfade_ramps[0])
        out_chunk = pad_or_trim(out_chunk, self.chunk_size_samples + preroll_samples)
        self.audio_history_ch1 = smooth_join(self.audio_history_ch1, out_chunk, *self.crossfade_ramps)
        self.audio_history_ch2 = np.concatenate((self.audio_history_ch2, audio_chunk), axis=-1)

        # Emit the output chunk from the audio history, shifted left by chunk_fade_secs.
        # This is done because the smooth join (crossfade) modifies the tail of the previous chunk.
        # So, we include the tail of the previous chunk in the output and exclude the tail of the current chunk.
        out_chunk = self.audio_history_ch1[-self.chunk_size_samples-self.crossfade_ramps[0]:-self.crossfade_ramps[0]]
        # In case of the first chunk there will be no previous chunk tail so pad it on the left with zeros
        out_chunk = pad_or_trim(out_chunk, self.chunk_size_samples, pad_side="left")

        # Sanity check - output size
        if out_chunk.shape[-1] != self.chunk_size_samples:
            raise ValueError(f"out_chunk must have length {self.chunk_size_samples}, but got {out_chunk.shape[-1]}")
        return out_chunk

    def append_transcript(self, text_input_ids: torch.LongTensor):
        if text_input_ids is None or text_input_ids.shape[-1] == 0:
            return
        text_str = self.resources.tokenizer.decode(text_input_ids[0], skip_special_tokens=False)
        text_str = text_str.replace(self.config.start_audio_token, "").strip()
        speaker = ""
        time_secs = (datetime.now() - self.start_time).total_seconds()
        if len(text_str) > 1 and text_str[1] == ":":
            speaker = text_str[0]
            text_str = text_str[2:].lstrip()
        self.transcript.append({
            "speaker": speaker,
            "text": text_str,
            "time_secs": time_secs,
        })

    def get_sequence_str(self) -> Tuple[str, str]:
        agent_sequence = self.resources.tokenizer.decode(self.input_ids[0], skip_special_tokens=False)
        return agent_sequence
    
    def get_audio_history(self) -> np.ndarray:
        audio_history = np.stack([self.audio_history_ch1, self.audio_history_ch2], axis=0)
        return audio_history
    
    def format_transcript(self) -> str:
        formatted_transcript = "\n".join([
            f"[{timedelta(seconds=int(entry['time_secs']))}] {entry['speaker']}: {entry['text']}" for entry in self.transcript
        ])
        return formatted_transcript