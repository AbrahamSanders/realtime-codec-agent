import numpy as np
import torch
import os
import re
import time
from dataclasses import dataclass
from warnings import warn
from datetime import timedelta, datetime
from typing import Union, Optional
from transformers import AutoTokenizer
from realtime_codec_agent.utils.llamacpp_utils import LlamaForAlternatingCodeChannels

from .audio_tokenizer import AudioTokenizer
from .utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim
from .realtime_agent_config import RealtimeAgentConfig
from .realtime_agent_profiler import RealtimeAgentProfilerCollection

class RealtimeAgentResources:
    def __init__(
        self, 
        llm_model_path: str = "Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-F16.gguf", 
        codec_model: str = "MagiCodec-50Hz-Base", 
        codec_device: Optional[Union[str, torch.device]] = None,
        whisper_model: Optional[str] = "small.en-q8_0",
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
        self.whisper_model = whisper_model
        if self.whisper_model is not None:
            from pywhispercpp.model import Model
            self.whisper_model = Model(self.whisper_model)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)

class RealtimeAgent:
    def __init__(self, resources: RealtimeAgentResources = None, config: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        self.resources = resources

        if config is None:
            config = RealtimeAgentConfig()
        self.set_config(config)

        self.transcript_regex = re.compile("([A-Z]):(.*?)(?= [A-Z]:|$)")

        self.reset()

    def reset(self):
        self.resources.audio_tokenizer.reset_context()
        c = self.config

        self.set_sampler()
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
        self.input_ids_history = self.input_ids.clone()
        self.trim_pos = self.input_ids.shape[-1]
        if c.agent_opening_text:
            agent_prompt += f" {c.agent_identity}: {c.agent_opening_text}"
        agent_prompt += c.start_audio_token
        self.input_ids = self.resources.tokenizer(agent_prompt, return_tensors="pt").input_ids
        self.resources.llm.eval(self.input_ids[0, :-1].tolist())

        self.generated_tokens = 0
        self.generated_audio_tokens = 0
        self.generated_audio_tokens_at_last_trans = 0
        self.audio_history_ch1, self.audio_history_ch2 = np.zeros((2, 0), dtype=np.float32)
        self.transcript = []
        if c.agent_opening_text:
            self.transcript.append({
                "speaker": c.agent_identity,
                "text": c.agent_opening_text,
                "time_secs": 0.0,
            })

        self.profilers.reset()

    def set_config(self, config: RealtimeAgentConfig):
        self.config = config
        
        self.chunk_size_samples = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.sampling_rate)
        self.chunk_size_frames = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.framerate * 2)

        self.max_context_frames = int(self.config.max_context_secs * self.resources.audio_tokenizer.framerate * 2)
        self.trim_by_frames = int(self.config.trim_by_secs * self.resources.audio_tokenizer.framerate * 2)

        self.force_trans_margin_frames = int(self.config.force_trans_margin_secs * self.resources.audio_tokenizer.framerate * 2)

        self.crossfade_ramps = create_crossfade_ramps(self.resources.audio_tokenizer.sampling_rate, fade_secs=self.config.chunk_fade_secs)

        self.end_header_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_header_token)
        self.start_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.start_audio_token)
        self.end_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_audio_token)
        self.agent_speaker_token_id = self.resources.tokenizer.encode(f" {self.config.agent_identity}", add_special_tokens=False)[0]
        self.user_speaker_token_id = self.resources.tokenizer.encode(f" {self.config.user_identity}", add_special_tokens=False)[0]

        self.profilers = RealtimeAgentProfilerCollection(self.config)

    def set_sampler(self, for_trans: bool = False) -> None:
        c = self.config
        self.resources.llm.init_sampler_for_generate(
            top_k=c.top_k,
            top_p=c.top_p,
            min_p=c.min_p,
            temp=c.trans_temperature if for_trans else c.temperature,
            repeat_penalty=c.repeat_penalty,
            presence_penalty=c.presence_penalty,
            frequency_penalty=c.frequency_penalty,
            seed=c.seed,
        )        

    def trim_sequences(self) -> None:
        generated_frames = self.generated_audio_tokens * 2
        if generated_frames >= self.max_context_frames and generated_frames % self.trim_by_frames == 0:
            audio_tokens_idx = torch.where(self.input_ids > self.end_header_token_id)[1]
            audio_tokens_idx = audio_tokens_idx[audio_tokens_idx >= self.trim_pos]
            trim_to_pos = audio_tokens_idx[self.trim_by_frames]
            # back up the input ids about to be trimmed
            self.input_ids_history = torch.cat([self.input_ids_history, self.input_ids[..., self.trim_pos:trim_to_pos]], dim=1)
            # trim the input ids
            self.input_ids = torch.cat(
                [
                    self.input_ids[..., :self.trim_pos],
                    self.input_ids[..., trim_to_pos:],
                ], 
                dim=1,
            )
            # recompute the kv cache for all remaining tokens after the header
            self.resources.llm.n_tokens = self.trim_pos
            audio_mode = (self.input_ids[0, -2:] > self.end_header_token_id).all()
            last_n = 2 if audio_mode else 1
            self.resources.llm.eval(self.input_ids[0, self.trim_pos:-last_n].tolist())

    def process_audio_input_ids(self, audio_chunk_input_ids: torch.LongTensor, force_trans: bool) -> torch.LongTensor:
        out_chunk_input_ids = torch.zeros((1, 0), dtype=audio_chunk_input_ids.dtype)
        for i in range(audio_chunk_input_ids.shape[-1]):
            # trim the sequences to the maximum length
            self.trim_sequences()
            text_start_pos = None
            while True:
                # predict next token
                audio_mode = (self.input_ids[0, -2:] > self.end_header_token_id).all()
                last_n = 2 if audio_mode else 1
                if force_trans:
                    self.resources.llm.eval(self.input_ids[0, -last_n:].tolist())
                    next_token = torch.LongTensor([[self.end_audio_token_id if audio_mode else self.user_speaker_token_id]])
                    force_trans = audio_mode
                else:
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
                # if next token is end_audio_token, note the position
                elif next_token[0, 0] == self.end_audio_token_id:
                    text_start_pos = self.input_ids.shape[-1]
                # if the previous token was end_audio_token and next token is *not* the agent speaker token,
                # set the sampler for transcription
                elif self.input_ids[0, -2] == self.end_audio_token_id and next_token[0, 0] != self.agent_speaker_token_id:
                    trans_input_ids = None
                    if self.config.use_whisper:
                        trans_input_ids = self.whisper_trans()
                    if trans_input_ids is not None:
                        self.input_ids = torch.cat([self.input_ids, trans_input_ids, torch.LongTensor([[self.start_audio_token_id]])], dim=1)
                        self.update_transcript(text_start_pos)
                        self.resources.llm.eval(self.input_ids[0, text_start_pos:-1].tolist())
                    else:
                        self.set_sampler(for_trans=True)
                    self.generated_audio_tokens_at_last_trans = self.generated_audio_tokens
                # if next token is start_audio_token, update the transcript and reset the sampler for audio generation
                elif next_token[0, 0] == self.start_audio_token_id:
                    self.update_transcript(text_start_pos)
                    self.set_sampler(for_trans=False)
                    
        return out_chunk_input_ids

    def whisper_trans(self):
        if self.resources.whisper_model is None:
            raise ValueError("Whisper model is not loaded.")
        trans_audio_start_secs = self.generated_audio_tokens_at_last_trans / self.resources.audio_tokenizer.framerate
        trans_audio_start_samples = int(trans_audio_start_secs * self.resources.audio_tokenizer.sampling_rate)
        trans_audio = self.audio_history_ch2[trans_audio_start_samples:]
        segments = self.resources.whisper_model.transcribe(
            trans_audio, 
            temperature=self.config.trans_temperature,
            language="en",
            no_context=True, 
            single_segment=True, 
            print_progress=False,
        )
        transcription = " ".join([segment.text for segment in segments])
        transcription = transcription.lower().replace("[blank_audio]", "").replace("...", "").replace(".", "").replace(">>", "").strip()
        if not transcription:
            return None
        transcription = f": {transcription}"
        trans_input_ids = self.resources.tokenizer(transcription, add_special_tokens=False, return_tensors="pt").input_ids
        return trans_input_ids

    def should_force_transcription(self, audio_chunk: np.ndarray) -> bool:
        if not self.config.force_trans_after_activity:
            return False
        last_in_chunk = self.audio_history_ch2[-self.chunk_size_samples:]
        last_in_chunk_abs_max = 0.0 if last_in_chunk.size == 0 else np.abs(last_in_chunk).max()
        curr_in_chunk_abs_max = np.abs(audio_chunk).max()

        activity_abs_max_threshold = 100 / 32768.0
        generated_frames = self.generated_audio_tokens * 2
        generated_frames_at_last_trans = self.generated_audio_tokens_at_last_trans * 2
        force_trans = last_in_chunk_abs_max >= activity_abs_max_threshold \
            and curr_in_chunk_abs_max < activity_abs_max_threshold \
            and (generated_frames - generated_frames_at_last_trans) > self.force_trans_margin_frames
        return force_trans

    def process_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        with self.profilers.total_profiler:
            # Sanity check - input size
            if audio_chunk.shape[-1] != self.chunk_size_samples:
                raise ValueError(f"audio_chunk must have length {self.chunk_size_samples}, but got {audio_chunk.shape[-1]}")

            # Encode audio chunk to input ids
            with self.profilers.audio_tokenize_profiler:
                audio_chunk_str = self.resources.audio_tokenizer.tokenize_audio(audio_chunk)
            with self.profilers.tokenize_profiler:
                audio_chunk_input_ids = self.resources.tokenizer(audio_chunk_str, add_special_tokens=False, return_tensors="pt").input_ids
            
            # Generate next audio chunk (interleaved by frame with the input audio chunk) and also generate any interleaved text
            with self.profilers.lm_profiler:
                force_trans = self.should_force_transcription(audio_chunk)
                out_chunk_input_ids = self.process_audio_input_ids(audio_chunk_input_ids, force_trans)

            # Decode input ids to audio chunk and append to the audio history
            with self.profilers.detokenize_profiler:
                out_chunk_str = self.resources.tokenizer.decode(out_chunk_input_ids[0], skip_special_tokens=False)
            with self.profilers.audio_detokenize_profiler:
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

    def update_transcript(self, text_start_pos: int) -> None:
        if text_start_pos is None:
            warn(
                "No text start position found, skipping transcript update. "
                "This means that the model did not previously generate a end_audio_token. "
                "This should never happen - investigate this!"
            )
            return
        text_str = self.resources.tokenizer.decode(self.input_ids[0, text_start_pos:-1], skip_special_tokens=False)
        # It is possible for there to be multiple speaker utterances in the same text chunk, so we extract them with regex
        transcript_entries = []
        for speaker, sp_text in self.transcript_regex.findall(text_str):
            sp_text = sp_text.lstrip()
            time_secs = self.generated_audio_tokens / self.resources.audio_tokenizer.framerate
            # If the speaker is not the agent, the text is a transcription in audio-first interleave mode and
            # we need to subtract the utterance length from time_secs.
            if speaker != self.config.agent_identity:
                pass
            transcript_entries.append({
                "speaker": speaker,
                "text": sp_text,
                "time_secs": time_secs,
            })
        self.transcript.extend(
            sorted(transcript_entries, key=lambda x: x["time_secs"])
        )

    def get_sequence_str(self, full: bool = False) -> str:
        if full:
            input_ids = torch.cat([self.input_ids_history, self.input_ids[..., self.trim_pos:]], dim=1)
        else:
            input_ids = self.input_ids
        agent_sequence = self.resources.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return agent_sequence
    
    def get_audio_history(self) -> np.ndarray:
        audio_history = np.stack([self.audio_history_ch1, self.audio_history_ch2], axis=0)
        return audio_history
    
    def format_transcript(self) -> str:
        formatted_transcript = "\n".join([
            f"[{timedelta(seconds=int(entry['time_secs']))}] {entry['speaker']}: {entry['text']}" for entry in self.transcript
        ])
        return formatted_transcript

@dataclass
class RealtimeAgentMultiprocessingInfo:
    config: RealtimeAgentConfig
    sampling_rate: int
    chunk_size_samples: int
    transcript: str
    sequence: str
    audio_history: np.ndarray

class RealtimeAgentMultiprocessing:
    def __init__(
        self, 
        wait_until_running: bool = True,
        config: RealtimeAgentConfig = None,
        idle_tol_secs: float = 1.0,
        **resources_kwargs,
    ):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.info_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)
        self.set_config_flag = ctx.Value(c_bool, False)
        self.reset_flag = ctx.Value(c_bool, False)
        self.get_info_flag = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(
            target=self.execute, 
            daemon=True, 
            args=(config, idle_tol_secs),
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

    def execute(self, config: RealtimeAgentConfig, idle_tol_secs: float, **resources_kwargs):
        agent_resources = RealtimeAgentResources(**resources_kwargs)
        agent = RealtimeAgent(resources=agent_resources, config=config)
        last_input_time = datetime.now()
        is_idle = False

        self.running.value = True
        print(">>> Agent is running! <<<")
        while True:
            try:
                if self.set_config_flag.value:
                    self.reset_flag.value = True
                    config = self.config_queue.get()
                    agent.set_config(config)
                    self.set_config_flag.value = False
                    print(">>> Config updated! <<<")

                if self.reset_flag.value:
                    agent.reset()
                    self._skip_queue(self.input_queue)
                    self.reset_flag.value = False
                    print(">>> Agent reset! <<<")

                if self.get_info_flag.value:
                    info = RealtimeAgentMultiprocessingInfo(
                        config=agent.config,
                        sampling_rate=agent.resources.audio_tokenizer.sampling_rate,
                        chunk_size_samples=agent.chunk_size_samples,
                        transcript=agent.format_transcript(),
                        sequence=agent.get_sequence_str(full=True),
                        audio_history=agent.get_audio_history(),
                    )
                    self.info_queue.put(info)
                    self.get_info_flag.value = False
                    print(">>> Info retrieved! <<<")

                now = datetime.now()
                if not self.input_queue.empty():
                    input_audio = self.input_queue.get()
                    output_audio = agent.process_audio(input_audio)
                    realtime_factor = agent.profilers.total_profiler.realtime_factor_values[-1] \
                        if agent.profilers.total_profiler.realtime_factor_values else None
                    self.output_queue.put((output_audio, realtime_factor))
                    if is_idle:
                        print(">>> Agent is no longer idle! <<<")
                    last_input_time = now
                    is_idle = False
                elif not is_idle:
                    secs_since_last_input = (now - last_input_time).total_seconds()
                    if secs_since_last_input >= idle_tol_secs:
                        print(">>> Agent is idle! <<<")
                        is_idle = True
            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            
            if is_idle:
                time.sleep(0.05)

    def _skip_queue(self, queue):
        val = None
        while not queue.empty():
            val = queue.get()
        return val

    def reset(self):
        self.reset_flag.value = True
        #TODO: use an Event instead of a loop
        while self.reset_flag.value:
            time.sleep(0.01)

    def set_config_and_reset(self, config):
        self.set_config_flag.value = True
        self.config_queue.put(config)
        #TODO: use an Event instead of a loop
        while self.set_config_flag.value or self.reset_flag.value:
            time.sleep(0.01)

    def get_info(self) -> RealtimeAgentMultiprocessingInfo:
        self.get_info_flag.value = True
        return self.info_queue.get()

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        if self.output_queue.empty():
            return None
        return self.output_queue.get()
    