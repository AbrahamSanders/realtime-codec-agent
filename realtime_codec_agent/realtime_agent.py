import numpy as np
import torch
import time
import itertools
from datetime import datetime, timedelta
from typing import Tuple, Union, List, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from dataclasses import dataclass

from .audio_tokenizer import AudioTokenizer
from .utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim, normalize_audio_rms
from .external_llm import get_external_llm_messages


class RealtimeAgentResources:
    def __init__(
        self, 
        vllm_model: str = "Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo", 
        codec_model: str = "MagiCodec-50Hz-Base", 
        device: Union[str, torch.device] = None,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.7,
        debug: bool = False,
        mock: bool = False,
    ):
        if mock:
            self.llm = None
        else:
            self.llm = LLM(
                model=vllm_model, 
                tensor_parallel_size=tensor_parallel_size, 
                gpu_memory_utilization=gpu_memory_utilization, 
                enable_prefix_caching=True, 
                enforce_eager=debug,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_model)
        self.audio_tokenizer = AudioTokenizer(codec_model=codec_model, device=device)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)

AUDIO_FIRST_TRANSCRIPTION_MODES = [
    "force_after_activity",  # always attempt to after last chunk with audio activity
    "predict", # predict if transcription should be done after every audio token (expensive)
    "none", # never attempt to transcribe audio - transcriptions are provided externally
]

@dataclass
class RealtimeAgentConfig:
    agent_opening_text: str = None
    agent_voice_enrollment: Tuple[int, np.ndarray] = None
    agent_voice_enrollment_text: str = None
    agent_identity: str = "A"
    user_identity: str = "B"
    text_first_audio_temperature: float = 1.0
    text_first_audio_top_p: float = 1.0
    text_first_audio_min_p: float = 0.002
    text_first_audio_presence_penalty: float = 0.0
    text_first_audio_frequency_penalty: float = 0.0
    text_first_text_temperature: float = 0.8
    audio_first_trans_mode: str = "force_after_activity"
    audio_first_cont_temperature: float = 0.6 
    audio_first_trans_temperature: float = 0.2
    chunk_size_secs: float = 0.1
    chunk_fade_secs: float = 0.02
    max_context_secs: float = 40.0
    trim_by_secs: float = 10.0
    target_volume_rms: float = 0.05
    non_activity_limit_secs: float = 3.0
    seed: Optional[int] = None
    audio_first_token: str = "<|audio_first|>"
    text_first_token: str = "<|text_first|>"
    header_speaker_token: str = "<|speaker|>"
    end_header_token: str = "<|end_header|>"
    start_audio_token: str = "<|audio|>"
    end_audio_token: str = "<|end_audio|>"
    use_external_llm: bool = False
    external_llm_api_key: str = None
    external_llm_base_url: str = None
    external_llm_model: str = "gpt-4o-mini"
    external_llm_top_p: float = 0.9
    external_llm_instructions: str = None

    def __post_init__(self):
        if int(self.chunk_size_secs*100) % 2 != 0:
            raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
        if self.chunk_fade_secs > self.chunk_size_secs:
            raise ValueError("Chunk fade length cannot be longer than the chunk size.")
        if self.agent_voice_enrollment is not None and not self.agent_voice_enrollment_text:
            raise ValueError("If agent_voice_enrollment is provided, agent_voice_enrollment_text must also be provided.")
        if self.audio_first_trans_mode not in AUDIO_FIRST_TRANSCRIPTION_MODES:
            raise ValueError(f"audio_first_trans_mode must be one of {AUDIO_FIRST_TRANSCRIPTION_MODES}.")

@dataclass
class GenerateParams:
    input_ids: torch.LongTensor
    sampling_params: SamplingParams

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
        common_header = f"{c.header_speaker_token}{c.agent_identity}{c.header_speaker_token}{c.user_identity}{c.end_header_token}"
        audio_first_prompt = f"{c.audio_first_token}{common_header}"
        text_first_prompt = f"{c.text_first_token}{common_header}"

        self.audio_first_input_ids = self.resources.tokenizer(audio_first_prompt, return_tensors="pt").input_ids
        self.text_first_input_ids = self.resources.tokenizer(text_first_prompt, return_tensors="pt").input_ids
        self.audio_first_trim_pos = self.audio_first_input_ids.shape[-1]
        self.text_first_trim_pos = self.text_first_input_ids.shape[-1]

        if c.agent_voice_enrollment is not None:
            silence_audio = (c.agent_voice_enrollment[0], np.zeros_like(c.agent_voice_enrollment[1]))
            silence_audio_str = self.resources.audio_tokenizer.chunked_tokenize_audio(silence_audio, c.chunk_size_secs)
            enrollment_audio_str = self.resources.audio_tokenizer.chunked_tokenize_audio(c.agent_voice_enrollment, c.chunk_size_secs)
            enrollment_audio_str = "".join(list(itertools.chain.from_iterable(zip(*[enrollment_audio_str, silence_audio_str]))))

            audio_first_prompt += f"{c.start_audio_token}{enrollment_audio_str}{c.end_audio_token} {c.agent_identity}: {c.agent_voice_enrollment_text}{c.start_audio_token}"
            text_first_prompt +=  f" {c.agent_identity}: {c.agent_voice_enrollment_text}{c.start_audio_token}{enrollment_audio_str}"

        if c.agent_opening_text:
            text_first_prompt += "" if text_first_prompt.endswith(c.end_header_token) else c.end_audio_token
            text_first_prompt += f" {c.agent_identity}: {c.agent_opening_text}{c.start_audio_token}"

        if audio_first_prompt.endswith(c.end_header_token):
            audio_first_prompt += c.start_audio_token
        if text_first_prompt.endswith(c.end_header_token):
            text_first_prompt += c.start_audio_token

        self.audio_first_input_ids = self.resources.tokenizer(audio_first_prompt, return_tensors="pt").input_ids
        self.text_first_input_ids = self.resources.tokenizer(text_first_prompt, return_tensors="pt").input_ids
        self.text_first_trans_pos = self.text_first_input_ids.shape[-1]-1 \
            if text_first_prompt.endswith(c.start_audio_token) else self.text_first_input_ids.shape[-1]

        self.generated_tokens = 0
        self.generated_audio_tokens = 0
        self.non_activity_elapsed_secs = 0.0
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
        self.user_speaker_token_id = self.resources.tokenizer.encode(f" {self.config.user_identity}", add_special_tokens=False)[0]

        if self.config.use_external_llm:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.external_llm_api_key, 
                base_url=self.config.external_llm_base_url,
            )

    def trim_sequences(self) -> None:
        if self.generated_audio_tokens > 0 and (self.generated_audio_tokens * 2) % self.max_context_frames == 0:
            audio_tokens_idx = torch.where(self.audio_first_input_ids > self.end_header_token_id)[1]
            trim_to_pos = audio_tokens_idx[self.trim_by_frames]
            self.audio_first_input_ids = torch.cat(
                [
                    self.audio_first_input_ids[..., :self.audio_first_trim_pos],
                    self.audio_first_input_ids[..., trim_to_pos:],
                ], 
                dim=1,
            )
            audio_tokens_idx = torch.where(self.text_first_input_ids > self.end_header_token_id)[1]
            trim_to_pos = audio_tokens_idx[self.trim_by_frames]
            self.text_first_input_ids = torch.cat(
                [
                    self.text_first_input_ids[..., :self.text_first_trim_pos],
                    self.text_first_input_ids[..., trim_to_pos:],
                ], 
                dim=1,
            )
            num_trimmed_tokens = trim_to_pos - self.text_first_trim_pos
            self.text_first_trans_pos = max(self.text_first_trim_pos, self.text_first_trans_pos-num_trimmed_tokens)

    def call_external_llm(self, text_first_next_tokens: torch.LongTensor) -> torch.LongTensor:
        if text_first_next_tokens.shape[-1] <= 6:
            return text_first_next_tokens

        messages = get_external_llm_messages(
            self.client.base_url.host, 
            self.config.external_llm_instructions, 
            self.transcript, 
            self.config.agent_identity,
        )
        response = self.client.chat.completions.create(
            model=self.config.external_llm_model,
            messages=messages,
            max_tokens=100,
            top_p=self.config.external_llm_top_p,
        )
        response_text = response.choices[0].message.content.replace("\n", " ").strip().lower()
        if response_text == "[silence]":
            return torch.LongTensor([[]])
        response_text = f" {self.config.agent_identity}: {response_text}{self.config.start_audio_token}"
        response_tokens = self.resources.tokenizer(response_text, add_special_tokens=False, return_tensors="pt").input_ids
        return response_tokens

    def generate(self, generate_params: Union[List[GenerateParams], GenerateParams]) -> Union[List[torch.LongTensor], torch.LongTensor]:
        if isinstance(generate_params, GenerateParams):
            generate_params = [generate_params]
        prompts = [TokensPrompt(prompt_token_ids=params.input_ids[0].tolist()) for params in generate_params]
        sampling_params = [params.sampling_params for params in generate_params]
        for params in sampling_params:
            params.skip_special_tokens = False
            params.spaces_between_special_tokens = False
            params.seed = self.config.seed
            params.detokenize = bool(params.stop)
        outputs = self.resources.llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        next_token_ids = [torch.tensor(output.outputs[0].token_ids).unsqueeze(0) for output in outputs]
        if len(next_token_ids) == 1:
            return next_token_ids[0]
        return next_token_ids

    def generate_for_mode(self, mode: str, force_trans: bool, force_response: bool) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor], int]:
        # TODO: clean up this awful mess
        max_tokens = 100 if "text" in mode else 1
        if mode == "audio" or mode == "force_audio":
            logit_bias = {self.end_audio_token_id: -100} if mode == "force_audio" else None
            if force_trans or force_response:
                audio_first_next_tokens = torch.LongTensor([[self.end_audio_token_id if force_trans else self.start_audio_token_id]])
                text_first_next_tokens = torch.LongTensor([[self.end_audio_token_id if force_response else self.start_audio_token_id]])
                generated_tokens = 0
            elif self.config.audio_first_trans_mode == "predict":
                audio_first_next_tokens, text_first_next_tokens = self.generate(
                    [
                        GenerateParams(
                            self.audio_first_input_ids, 
                            SamplingParams(
                                max_tokens=max_tokens, 
                                temperature=self.config.audio_first_cont_temperature,
                            ),
                        ),
                        GenerateParams(
                            self.text_first_input_ids, 
                            SamplingParams(
                                max_tokens=max_tokens, 
                                temperature=self.config.text_first_audio_temperature, 
                                top_p=self.config.text_first_audio_top_p,
                                min_p=self.config.text_first_audio_min_p,
                                presence_penalty=self.config.text_first_audio_presence_penalty, 
                                frequency_penalty=self.config.text_first_audio_frequency_penalty, 
                                logit_bias=logit_bias,
                            ),
                        ),
                    ]
                )
                generated_tokens = audio_first_next_tokens.shape[-1] + text_first_next_tokens.shape[-1]
            else:
                audio_first_next_tokens = torch.LongTensor([[self.start_audio_token_id]])
                text_first_next_tokens = self.generate(
                    [
                        GenerateParams(
                            self.text_first_input_ids, 
                            SamplingParams(
                                max_tokens=max_tokens, 
                                temperature=self.config.text_first_audio_temperature, 
                                top_p=self.config.text_first_audio_top_p,
                                min_p=self.config.text_first_audio_min_p,
                                presence_penalty=self.config.text_first_audio_presence_penalty, 
                                frequency_penalty=self.config.text_first_audio_frequency_penalty, 
                                logit_bias=logit_bias,
                            ),
                        ),
                    ]
                )
                generated_tokens = text_first_next_tokens.shape[-1]
        elif mode == "both_text":
            audio_first_next_tokens, text_first_next_tokens = self.generate(
                [
                    GenerateParams(
                        self.audio_first_input_ids, 
                        SamplingParams(
                            max_tokens=max_tokens, 
                            temperature=self.config.audio_first_trans_temperature, 
                            stop_token_ids=[self.start_audio_token_id],
                        ),
                    ),
                    GenerateParams(
                        self.text_first_input_ids, 
                        SamplingParams(
                            max_tokens=max_tokens, 
                            temperature=self.config.text_first_text_temperature, 
                            stop_token_ids=[self.start_audio_token_id], 
                            stop=f" {self.config.user_identity}:",
                        ),
                    ),
                ]
            )
            generated_tokens = audio_first_next_tokens.shape[-1] + text_first_next_tokens.shape[-1]
        elif mode == "audio_first_text":
            audio_first_next_tokens = self.generate(
                GenerateParams(
                    self.audio_first_input_ids, 
                    SamplingParams(
                        max_tokens=max_tokens, 
                        temperature=self.config.audio_first_trans_temperature,
                        stop_token_ids=[self.start_audio_token_id],
                    ),
                ),
            )
            generated_tokens = audio_first_next_tokens.shape[-1]
            text_first_next_tokens = None
        elif mode == "text_first_text":
            text_first_next_tokens = self.generate(
                GenerateParams(
                    self.text_first_input_ids, 
                    SamplingParams(
                        max_tokens=max_tokens, 
                        temperature=self.config.text_first_text_temperature, 
                        stop_token_ids=[self.start_audio_token_id], 
                        stop=f" {self.config.user_identity}:",
                    ),
                ),
            )
            generated_tokens = text_first_next_tokens.shape[-1]
            audio_first_next_tokens = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if "text" in mode:
            if audio_first_next_tokens is not None and audio_first_next_tokens[0, -1] != self.start_audio_token_id:
                audio_first_next_tokens = torch.cat([audio_first_next_tokens, torch.LongTensor([[self.start_audio_token_id]])], dim=1)
            if text_first_next_tokens is not None and text_first_next_tokens[0, -1] != self.start_audio_token_id:
                text_first_next_tokens = torch.cat([text_first_next_tokens, torch.LongTensor([[self.start_audio_token_id]])], dim=1)

        return audio_first_next_tokens, text_first_next_tokens, generated_tokens

    def process_audio_input_ids(self, audio_chunk_input_ids: torch.LongTensor, force_trans: bool, force_response: bool) -> torch.LongTensor:
        out_chunk_input_ids = torch.zeros((1, 0), dtype=audio_chunk_input_ids.dtype)
        for i in range(audio_chunk_input_ids.shape[-1]):
            # trim the sequences to the maximum length
            self.trim_sequences()
            mode = "audio"
            while True:
                # predict next tokens
                audio_first_next_tokens, text_first_next_tokens, generated_tokens = self.generate_for_mode(mode, force_trans, force_response)
                force_trans = force_response = False
                self.generated_tokens += generated_tokens
                if mode == "audio" or mode == "force_audio":
                    if audio_first_next_tokens[0, 0] == self.end_audio_token_id and text_first_next_tokens[0, 0] == self.end_audio_token_id:
                        mode = "both_text"
                    elif audio_first_next_tokens[0, 0] == self.end_audio_token_id:
                        mode = "audio_first_text"
                    elif text_first_next_tokens[0, 0] == self.end_audio_token_id:
                        mode = "text_first_text"
                    else:
                        next_input_token = audio_chunk_input_ids[..., i:i+1]
                        self.audio_first_input_ids = torch.cat([self.audio_first_input_ids, text_first_next_tokens, next_input_token], dim=1)
                        self.text_first_input_ids = torch.cat([self.text_first_input_ids, text_first_next_tokens, next_input_token], dim=1)
                        out_chunk_input_ids = torch.cat([out_chunk_input_ids, text_first_next_tokens], dim=1)
                        self.generated_audio_tokens += 1
                        mode = "audio"
                        break # move on to next input token
                if mode == "audio_first_text" or mode == "both_text":
                    self.audio_first_input_ids = torch.cat([self.audio_first_input_ids, audio_first_next_tokens], dim=1)
                    if audio_first_next_tokens[0, 0] != self.end_audio_token_id:
                        if mode != "both_text":
                            mode = "audio"
                        # splice the transcription into the text-first sequence if it belongs to the user speaker
                        if audio_first_next_tokens[0, 0] == self.user_speaker_token_id:
                            add_control_tokens = self.text_first_input_ids[0, self.text_first_trans_pos] != self.start_audio_token_id
                            audio_start_id = torch.LongTensor([[self.start_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
                            audio_end_id = torch.LongTensor([[self.end_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
                            self.text_first_input_ids = torch.cat(
                                [
                                    self.text_first_input_ids[..., :self.text_first_trans_pos],
                                    audio_end_id, 
                                    audio_first_next_tokens[..., :-1],
                                    audio_start_id,
                                    self.text_first_input_ids[..., self.text_first_trans_pos:]
                                ], 
                                dim=1,
                            )
                            self.append_transcript(audio_first_next_tokens)
                        self.text_first_trans_pos = self.text_first_input_ids.shape[-1]-1 \
                            if self.text_first_input_ids[0, -1] == self.start_audio_token_id else self.text_first_input_ids.shape[-1]
                if mode == "text_first_text" or mode == "both_text":
                    if text_first_next_tokens[0, 0] == self.user_speaker_token_id:
                        # discard predictions for the user speaker on the text-first sequence
                        # and force the next token to be an audio token
                        self.text_first_input_ids = self.text_first_input_ids[..., :-1]
                        mode = "force_audio"
                    else:
                        if text_first_next_tokens[0, 0] != self.end_audio_token_id:
                            if self.config.use_external_llm:
                                text_first_next_tokens = self.call_external_llm(text_first_next_tokens)
                            if text_first_next_tokens.shape[-1] == 0:
                                self.text_first_input_ids = self.text_first_input_ids[..., :-1]
                                mode = "force_audio"
                            else:
                                self.append_transcript(text_first_next_tokens)
                                mode = "audio"
                        self.text_first_input_ids = torch.cat([self.text_first_input_ids, text_first_next_tokens], dim=1)
        return out_chunk_input_ids

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

    def inject_transcription_chunk(self, trans_chunk: Tuple[str, float]):
        if trans_chunk is None:
            return
        trans_chunk, trans_chunk_start = trans_chunk
        trans_chunk = trans_chunk.lower().replace(".", "").replace(",", "").strip()
        
        trans_start_frames = int(trans_chunk_start * self.resources.audio_tokenizer.framerate * 2)
        # compute the position to inject the trancription chunk
        audio_tokens_idx = torch.where(self.text_first_input_ids > self.end_header_token_id)[1]
        num_trimmed_frames = self.generated_audio_tokens * 2 - audio_tokens_idx.shape[-1]
        trans_start_frames -= num_trimmed_frames
        trans_pos = audio_tokens_idx[trans_start_frames] if trans_start_frames < audio_tokens_idx.shape[-1] else audio_tokens_idx[-1]
        if self.text_first_input_ids[0, trans_pos-1] == self.start_audio_token_id:
            trans_pos -= 1
        
        trans_chunk = f" {self.config.user_identity}: {trans_chunk}"
        print(f"Injecting: '{trans_chunk}'")
        trans_chunk_input_ids = self.resources.tokenizer(trans_chunk, add_special_tokens=False, return_tensors="pt").input_ids
        add_control_tokens = self.text_first_input_ids[0, trans_pos] != self.start_audio_token_id
        audio_start_id = torch.LongTensor([[self.start_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
        audio_end_id = torch.LongTensor([[self.end_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
        self.text_first_input_ids = torch.cat(
            [
                self.text_first_input_ids[..., :trans_pos],
                audio_end_id, 
                trans_chunk_input_ids,
                audio_start_id,
                self.text_first_input_ids[..., trans_pos:]
            ], 
            dim=1,
        )
        self.append_transcript(trans_chunk_input_ids)

    def should_force_text(self, audio_chunk: np.ndarray):
        last_in_chunk = self.audio_history_ch2[-self.chunk_size_samples:]
        last_out_chunk = self.audio_history_ch1[-2*self.chunk_size_samples:-self.chunk_size_samples]
        curr_in_chunk = audio_chunk
        curr_out_chunk = self.audio_history_ch1[-self.chunk_size_samples:]

        last_in_chunk_abs_max = 0.0 if last_in_chunk.size == 0 else np.abs(last_in_chunk).max()
        last_out_chunk_abs_max = 0.0 if last_out_chunk.size == 0 else np.abs(last_out_chunk).max()
        curr_in_chunk_abs_max = np.abs(curr_in_chunk).max()
        curr_out_chunk_abs_max = 0.0 if curr_out_chunk.size == 0 else np.abs(curr_out_chunk).max()

        activity_abs_max_threshold = 100 / 32768.0
        force_trans = force_response = False
        # Should force transcription?
        if self.config.audio_first_trans_mode == "force_after_activity":
            force_trans = (
                (last_in_chunk_abs_max >= activity_abs_max_threshold and curr_in_chunk_abs_max < activity_abs_max_threshold) or
                (last_out_chunk_abs_max >= activity_abs_max_threshold and curr_out_chunk_abs_max < activity_abs_max_threshold)
            )
        # Should force response?
        if curr_in_chunk_abs_max >= activity_abs_max_threshold or curr_out_chunk_abs_max >= activity_abs_max_threshold:
            self.non_activity_elapsed_secs = 0.0
        else:
            self.non_activity_elapsed_secs += self.config.chunk_size_secs
        if self.config.non_activity_limit_secs > 0:
            force_response = self.non_activity_elapsed_secs >= self.config.non_activity_limit_secs
        return force_trans, force_response

    def process_audio(self, audio_chunk: np.ndarray, trans_chunk: Optional[Tuple[str, float]] = None) -> np.ndarray:
        # Sanity check - input size
        if audio_chunk.shape[-1] != self.chunk_size_samples:
            raise ValueError(f"audio_chunk must have length {self.chunk_size_samples}, but got {audio_chunk.shape[-1]}")

        self.inject_transcription_chunk(trans_chunk)

        # Encode audio chunk to input ids
        audio_chunk_str = self.resources.audio_tokenizer.tokenize_audio(audio_chunk)
        audio_chunk_input_ids = self.resources.tokenizer(audio_chunk_str, add_special_tokens=False, return_tensors="pt").input_ids
        
        # Generate next audio chunk (interleaved by frame with the input audio chunk) and also generate any interleaved text
        if self.resources.llm is None:
            # Mock mode, just return the audio chunk as is
            out_chunk_input_ids = audio_chunk_input_ids
        else:
            force_trans, force_response = self.should_force_text(audio_chunk)
            out_chunk_input_ids = self.process_audio_input_ids(audio_chunk_input_ids, force_trans, force_response)

        # Decode input ids to audio chunk and append to the audio history
        out_chunk_str = self.resources.tokenizer.decode(out_chunk_input_ids[0], skip_special_tokens=False)
        (_, out_chunk), _, preroll_samples = self.resources.audio_tokenizer.detokenize_audio(out_chunk_str, preroll_samples=self.crossfade_ramps[0])
        out_chunk = pad_or_trim(out_chunk, self.chunk_size_samples + preroll_samples)
        if self.config.target_volume_rms > 0:
            out_chunk = normalize_audio_rms(out_chunk, target_rms=self.config.target_volume_rms)
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
    
    def get_sequence_strs(self) -> Tuple[str, str]:
        audio_first_sequence = self.resources.tokenizer.decode(self.audio_first_input_ids[0], skip_special_tokens=False)
        text_first_sequence = self.resources.tokenizer.decode(self.text_first_input_ids[0], skip_special_tokens=False)
        return audio_first_sequence, text_first_sequence
    
    def get_audio_history(self) -> np.ndarray:
        audio_history = np.stack([self.audio_history_ch1, self.audio_history_ch2], axis=0)
        return audio_history
    
    def format_transcript(self) -> str:
        formatted_transcript = "\n".join([
            f"[{timedelta(seconds=int(entry['time_secs']))}] {entry['speaker']}: {entry['text']}" for entry in self.transcript
        ])
        return formatted_transcript

class RealtimeAgentMultiprocessing:
    def __init__(
        self, 
        wait_until_running: bool = True,
        config: RealtimeAgentConfig = None,
        **resources_kwargs,
    ):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.sequence_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)
        self.paused = ctx.Value(c_bool, True)
        self.reset_flag = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(
            target=self.execute, 
            daemon=True, 
            args=(config,),
            kwargs=resources_kwargs,
        )
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            time.sleep(0.01)

    def is_running(self):
        return self.running.value
    
    def is_paused(self):
        return self.paused.value

    def execute(self, config: RealtimeAgentConfig, **resources_kwargs):
        agent_resources = RealtimeAgentResources(**resources_kwargs)
        agent = RealtimeAgent(resources=agent_resources, config=config)

        self.running.value = True
        print(">>> Agent is running! <<<")
        while True:
            try:
                new_config = self._skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config
                    agent.set_config(config)
                    self.reset()
                    print(">>> Config updated! <<<")

                if self.reset_flag.value:
                    agent.reset()
                    self._skip_queue(self.input_queue)
                    self.reset_flag.value = False
                    print(">>> Agent reset! <<<")

                if not self.is_paused() and not self.input_queue.empty():
                    input_audio = self.input_queue.get()
                    output_audio = agent.process_audio(input_audio)
                    self.output_queue.put(output_audio)

                    self.sequence_queue.put((
                        agent.audio_first_sequence,
                        agent.text_first_sequence,
                        agent.audio_history,
                    ))

            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            
            if self.is_paused():
                time.sleep(0.05)

    def _skip_queue(self, queue):
        val = None
        while not queue.empty():
            val = queue.get()
        return val

    def reset(self):
        self.reset_flag.value = True

    def pause(self):
        self.paused.value = True
        print(">>> Agent paused! <<<")

    def resume(self):
        self.paused.value = False
        print(">>> Agent resumed! <<<")

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        if self.output_queue.empty():
            return None
        return self.output_queue.get()
    
    def next_sequence(self):
        return self._skip_queue(self.sequence_queue)