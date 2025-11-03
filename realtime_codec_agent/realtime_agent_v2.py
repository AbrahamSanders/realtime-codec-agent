import numpy as np
import re
import time
from scipy.special import softmax
from dataclasses import dataclass
from warnings import warn
from datetime import datetime
from typing import Union, Optional, List, Dict, Any, Tuple, Set

from .utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim, normalize_audio_rms
from .realtime_agent_resources import RealtimeAgentResources
from .realtime_agent_config import RealtimeAgentConfig
from .realtime_agent_stats import RealtimeAgentStatsCollection
from .realtime_agent_profiler import RealtimeAgentProfilerCollection

class RealtimeAgent:
    def __init__(self, resources: RealtimeAgentResources = None, config: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        self.resources = resources

        self.llm_client = None
        self.tts_client = None
        if config is None:
            config = RealtimeAgentConfig()
        self.set_config(config)

        self.transcript_regex = re.compile("([A-Z]):(.*?)(?= [A-Z]:|$)")
        # Regex for constraining text generation to paralinguistics only (e.g. [laughing], &=laughs) when using external 
        # asr / llm instead of native transcription / response generation. Matches here indicate that generation should 
        # stop and the last token be discarded.
        self.constrained_text_gen_stop_regex = re.compile(r"\A(?:[^ ]| [^&[]| &[^=]| &=.* | \[.*\] )")
        # Wordlist of allowed backchannels and fillers when using external asr / llm instead of native transcription / response generation. 
        # Generation of a free-standing word (not in [...] or &=...) outside this list indicates that generation should stop and the last token be discarded.
        self.constrained_text_gen_wordlist = set(["yeah", "sure", "right", "okay", "well", "so", "and", "like", "you", "know", "uh", "huh", "um", "oh", "ah", "mm", "mmm", "hm", "hmm", "mhm", "mhmm"])

        self.reset()

    @property
    def total_frames(self) -> int:
        return len(self.audio_tokens_idx)

    @property
    def total_secs(self) -> float:
        return self.total_frames / (self.resources.audio_tokenizer.framerate * 2)

    @property
    def last_transcription(self) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.transcript):
            if entry["speaker"] != self.config.agent_identity:
                return entry
        return None
    
    @property
    def last_response(self) -> Optional[Dict[str, Any]]:
        for entry in reversed(self.transcript):
            if entry["speaker"] == self.config.agent_identity:
                return entry
        return None

    def reset(self):
        self.resources.audio_tokenizer.reset_context()
        c = self.config

        self.set_sampler()
        self.resources.llm.reset()
        if c.use_external_llm:
            self.llm_client.close_stream(blocking=True)
        if c.use_external_tts:
            self.tts_client.close_stream()
            self.tts_interrupted_chunk_input_ids = None
        
        voice_enrollment = np.zeros(self.resources.audio_tokenizer.sampling_rate * 3, dtype=np.float32) \
            if c.agent_voice_enrollment is None else c.agent_voice_enrollment
        enrollment_audio_str = self.resources.audio_tokenizer.chunked_tokenize_audio(voice_enrollment, c.chunk_size_secs)
        if c.use_external_tts:
            external_tts_prompt_text = c.external_tts_prompt_text.strip() if c.external_tts_prompt_text is not None else None
            if c.use_whisper and c.agent_voice_enrollment is not None and not external_tts_prompt_text:
                external_tts_prompt_text = self._whisper_trans(c.agent_voice_enrollment)
            self.tts_client.set_voice_enrollment(c.agent_voice_enrollment, external_tts_prompt_text)

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
        self.input_ids = self.resources.tokenizer.encode(agent_prompt)
        self.context_start_pos = len(self.input_ids)
        if c.agent_opening_text:
            agent_prompt += f" {c.agent_identity}: {c.agent_opening_text}"
        agent_prompt += c.start_audio_token
        self.input_ids = self.resources.tokenizer.encode(agent_prompt)
        self.resources.llm.eval(self.input_ids[:-1])

        self.trim_to_secs = 0.0
        self.ch1_inactivity_elapsed_secs = 0.0
        self.ch2_inactivity_elapsed_secs = 0.0
        self.ch2_activity_start_secs = 0.0
        self.audio_history_ch1: List[np.ndarray] = [] 
        self.audio_history_ch2: List[np.ndarray] = []
        self.audio_tokens_idx = []
        self.transcript = []
        if c.agent_opening_text:
            self.transcript.append({
                "speaker": c.agent_identity,
                "text": c.agent_opening_text,
                "start_secs": 0.0,
                "end_secs": None,
                "text_start_pos": self.context_start_pos,
                "text_with_external_markers": c.agent_opening_text,
            })
            if c.use_external_tts:
                self.tts_client.prep_stream(c.agent_opening_text)

        self.prob_event_speaker_token_id = None
        self.stats.reset()
        self.profilers.reset()

    def set_config(self, config: RealtimeAgentConfig):
        self.config = config
        
        self.chunk_size_samples = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.sampling_rate)
        self.chunk_size_frames_per_channel = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.framerate)
        self.crossfade_ramps = create_crossfade_ramps(self.resources.audio_tokenizer.sampling_rate, fade_secs=self.config.chunk_fade_secs)

        self.end_header_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_header_token)
        self.start_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.start_audio_token)
        self.end_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.end_audio_token)
        self.external_marker_token_id = self.resources.tokenizer.encode(self.config.external_marker_token, add_special_tokens=False)[0]
        self.agent_speaker_token_id = self.resources.tokenizer.encode(f" {self.config.agent_identity}", add_special_tokens=False)[0]
        self.user_speaker_token_id = self.resources.tokenizer.encode(f" {self.config.user_identity}", add_special_tokens=False)[0]

        if self.llm_client is not None:
            self.llm_client.close_stream(blocking=True)
        self.llm_client = None
        if self.config.use_external_llm:
            from .external_llm_client import ExternalLLMClient
            self.llm_client = ExternalLLMClient(
                api_key=self.config.external_llm_api_key, 
                base_url=self.config.external_llm_base_url,
                model=self.config.external_llm_model,
                agent_identity=self.config.agent_identity,
                allow_laughter=self.config.constrain_allow_laughter,
            )

        if self.tts_client is not None:
            self.tts_client.close_stream()
        self.tts_client = None
        if self.config.use_external_tts:
            from .external_tts_client import ExternalTTSClient
            from .external_tts_duplex_aligner import ExternalTTSDuplexAligner
            self.tts_client = ExternalTTSClient(
                server_url=self.config.external_tts_server_url, 
                chunk_size_secs=self.config.chunk_size_secs,
            )
            self.tts_duplex_aligner = ExternalTTSDuplexAligner(self.resources.audio_tokenizer, self.resources.llm_model_dir)
            if not self.config.external_tts_allow_fallback:
                self.resources.audio_tokenizer.reset_context()
                silence = np.zeros(self.resources.audio_tokenizer.context_samples, dtype=np.float32)
                self.default_tts_fallback_chunk = self.resources.audio_tokenizer.tokenize_audio(silence)[-self.chunk_size_frames_per_channel:]

        self.stats = RealtimeAgentStatsCollection(self.config)
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
        if self.total_secs - self.trim_to_secs >= self.config.max_context_secs:
            self.trim_to_secs += self.config.trim_by_secs
            self.recompute_kv_cache(0)

    def _native_generate_text(self, constrained: bool = False, allowed_wordlist: Optional[Set[str]] = None) -> int:
        text_start_pos = len(self.input_ids)
        text_start_n_tokens = self.resources.llm.n_tokens
        while True:
            next_token = next(self.resources.llm.generate(self.input_ids[-1:], reset=False))
            self.input_ids.append(next_token)
            if next_token == self.start_audio_token_id:
                break
            # if constrained we only allow paralinguistic content from the model
            if constrained:
                text = self.resources.tokenizer.decode(self.input_ids[text_start_pos:], skip_special_tokens=False).lower()
                if text == ":":
                    text_start_pos = len(self.input_ids)
                    text_start_n_tokens = self.resources.llm.n_tokens 
                elif re.match(self.constrained_text_gen_stop_regex, text) and (not allowed_wordlist or text.split()[-1] not in allowed_wordlist):
                    self.input_ids = self.input_ids[:-1]
                    self.resources.llm.n_tokens -= 1
                    break
        # Check for conditions requiring a rollback
        if constrained and len(self.input_ids) > text_start_pos:
            text = self.resources.tokenizer.decode(self.input_ids[text_start_pos:], skip_special_tokens=False).lower()
            if (
                (not self.config.constrain_allow_noise and ("noise" in text or "wind" in text or "blow" in text or "mn" in text)) or
                (not self.config.constrain_allow_breathing and ("breath" in text or "hh" in text or "cough" in text)) or
                (not self.config.constrain_allow_laughter and "laugh" in text)
            ):
                self.input_ids = self.input_ids[:text_start_pos]
                self.resources.llm.n_tokens = text_start_n_tokens
        return len(self.input_ids)-text_start_pos

    def _coordinated_generate_text(self) -> List[Tuple[int, int]]:
        external_pos_ranges = []
        next_response_sentence = self.llm_client.next_sentence()
        if next_response_sentence is None:
            self.llm_client.prep_stream(
                transcript=self.transcript,
                additional_instructions=self.config.external_llm_instructions,
                top_p=self.config.external_llm_top_p,
            )
            next_response_sentence = self.llm_client.next_sentence()
        if next_response_sentence is None or next_response_sentence.lower().startswith("[silen"):
            return external_pos_ranges
        ext_start_pos = len(self.input_ids)
        while True:
            next_response_sentence = next_response_sentence.lower().replace(",", "").replace(".", "")
            next_response_sentence = f" {next_response_sentence}"
            next_response_input_ids = self.resources.tokenizer.encode(next_response_sentence, add_special_tokens=False)
            self.input_ids.extend(next_response_input_ids)
            self.resources.llm.eval(self.input_ids[-len(next_response_input_ids)-1:-1])
            n_native = self._native_generate_text(constrained=True, allowed_wordlist=self.constrained_text_gen_wordlist)
            if n_native > 0:
                ext_end_pos = len(self.input_ids)-n_native
                external_pos_ranges.append((ext_start_pos, ext_end_pos))
                ext_start_pos = len(self.input_ids)
            if self.input_ids[-1] == self.start_audio_token_id:
                break
            next_response_sentence = self.llm_client.next_sentence()
            if next_response_sentence is None:
                ext_end_pos = len(self.input_ids)
                if ext_end_pos > ext_start_pos:
                    external_pos_ranges.append((ext_start_pos, ext_end_pos))
                break
        return external_pos_ranges

    def _complete_or_rollback_generate(self, text_start_pos: int, text_start_n_tokens: int, external_pos_ranges: List[Tuple[int, int]]) -> bool:
        # If nothing was generated, suppress the generation by removing the end_audio_token_id and speaker id.
        # Otherwise end with start_audio_token_id and update the transcript.
        if len(self.input_ids)-text_start_pos < 2:
            self.input_ids = self.input_ids[:text_start_pos-2]
            self.resources.llm.n_tokens = text_start_n_tokens-3
            return False
        else:
            if self.input_ids[-1] != self.start_audio_token_id:
                self.resources.llm.eval(self.input_ids[-1:])
                self.input_ids.append(self.start_audio_token_id)
            self.update_transcript(text_start_pos-1, external_pos_ranges)
            return True

    def generate_for_trans(self) -> None:
        assert self.input_ids[-2] == self.end_audio_token_id and self.input_ids[-1] != self.agent_speaker_token_id, \
            "generate_for_trans can only be called if self.input_ids ends with end_audio_token_id followed by a non-agent speaker id."
        # Record state at start in case we need to undo (e.g. if transcription is suppressed)
        text_start_pos = len(self.input_ids)
        text_start_n_tokens = self.resources.llm.n_tokens
        # Generate natively
        self.set_sampler(for_trans=True)
        self._native_generate_text(constrained=self.config.use_whisper)
        # If using whisper, call it now
        external_pos_ranges = []
        if self.config.use_whisper:
            trans_input_ids = self.whisper_trans()
            if trans_input_ids:
                if self.input_ids[-1] == self.start_audio_token_id:
                    self.input_ids = self.input_ids[:-1]
                else:
                    self.resources.llm.eval(self.input_ids[-1:])
                ext_start_pos = len(self.input_ids)
                self.input_ids.extend(trans_input_ids)
                ext_end_pos = len(self.input_ids)
                self.resources.llm.eval(self.input_ids[ext_start_pos:ext_end_pos-1])
                external_pos_ranges.append((ext_start_pos, ext_end_pos))
                # Give native lm a chance to add any trailing paralinguistic tokens
                self._native_generate_text(constrained=True, allowed_wordlist=self.constrained_text_gen_wordlist)
        self.set_sampler()
        completed = self._complete_or_rollback_generate(text_start_pos, text_start_n_tokens, external_pos_ranges)
        if completed and self.config.use_external_llm:
            # Prepare the external llm stream in advance to reduce latency
            self.llm_client.prep_stream(
                transcript=self.transcript,
                additional_instructions=self.config.external_llm_instructions,
                top_p=self.config.external_llm_top_p,
            )
        elif not completed:
            # since the transcription was suppressed, the transcription probability could still be high.
            # reset the channel 2 inactivity timer to avoid immediate forced transcription again.
            self.ch2_inactivity_elapsed_secs = 0.0

    def generate_for_response(self) -> None:
        assert self.input_ids[-2] == self.end_audio_token_id and self.input_ids[-1] == self.agent_speaker_token_id, \
            "generate_for_response can only be called if self.input_ids ends with end_audio_token_id followed by the agent speaker id."
        self.finalize_last_response()
        # Record state at start in case we need to undo (e.g. if response is suppressed)
        text_start_pos = len(self.input_ids)
        text_start_n_tokens = self.resources.llm.n_tokens
        # Generate natively
        allowed_wordlist = self.constrained_text_gen_wordlist \
            if self.config.use_external_llm and (self.llm_client.stream is None or self.llm_client.stream_read_count == 0) \
            else None
        self._native_generate_text(constrained=self.config.use_external_llm, allowed_wordlist=allowed_wordlist)
        # If using external LLM, call it now
        external_pos_ranges = []
        if self.config.use_external_llm and self.input_ids[-1] != self.start_audio_token_id and self.prob_event_speaker_token_id != self.user_speaker_token_id:
            external_pos_ranges = self._coordinated_generate_text()
        self._complete_or_rollback_generate(text_start_pos, text_start_n_tokens, external_pos_ranges)
        # the model has the intent to respond, so reset channel 1 inactivity timer even though its audio hasn't been generated yet.
        # this prevents duplicate responses when self.config.force_response_after_inactivity_secs > 0.0.
        self.ch1_inactivity_elapsed_secs = 0.0

    def process_audio_input_ids(
        self, 
        audio_chunk_input_ids: Optional[List[int]] = None, 
        force_trans: bool = False, 
        force_response: bool = False, 
    ) -> List[int]:
        if audio_chunk_input_ids is None:
            audio_chunk_input_ids = [None]
        out_chunk_input_ids = [0] * len(audio_chunk_input_ids)
        for i in range(len(audio_chunk_input_ids)):
            # trim the sequences to the maximum length
            self.trim_sequences()
            while True:
                # predict next token
                audio_mode = all([t > self.end_header_token_id for t in self.input_ids[-2:]])
                if audio_mode and (force_trans or force_response):
                    self.input_ids.append(self.end_audio_token_id)
                    self.resources.llm.eval(self.input_ids[-3:])
                    next_token = self.user_speaker_token_id if force_trans else self.agent_speaker_token_id
                    force_trans = force_response = False
                else:
                    last_n = 2 if audio_mode else 1
                    next_token = next(self.resources.llm.generate(self.input_ids[-last_n:], reset=False))
                self.input_ids.append(next_token)
                # if next token is an audio token, append the next input (user) audio token
                if next_token > self.end_header_token_id:
                    if audio_chunk_input_ids[i] is not None:
                        self.input_ids.append(audio_chunk_input_ids[i])
                        self.audio_tokens_idx.extend([len(self.input_ids)-2, len(self.input_ids)-1])
                    else:
                        self.audio_tokens_idx.append(len(self.input_ids)-1)
                    out_chunk_input_ids[i] = next_token
                    break # move on to next input token
                # if the previous token was end_audio_token and next token is *not* the agent speaker token, do transcription
                elif self.input_ids[-2] == self.end_audio_token_id and next_token != self.agent_speaker_token_id:
                    self.generate_for_trans()
                # if the previous token was end_audio_token and next token *is* the agent speaker token, do response generation
                elif self.input_ids[-2] == self.end_audio_token_id and next_token == self.agent_speaker_token_id:
                    self.generate_for_response()
        return out_chunk_input_ids
    
    def process_tts_input_ids(
        self, 
        tts_chunk_input_ids: List[int], 
        out_chunk_input_ids: List[int], 
    ) -> List[int]:
        if tts_chunk_input_ids is None:
            return out_chunk_input_ids
        # interrupt the external tts stream if the duplex lm's agent response predictions are diverging toward silence
        tts_interrupt_score = self.tts_duplex_aligner.interrupt_score(tts_chunk_input_ids, out_chunk_input_ids)
        #tts_interrupt_score = np.clip(tts_interrupt_score, 0.0, 20.0)
        self.stats.tts_interrupt_score.add_value(tts_interrupt_score)
        if self.stats.tts_interrupt_score.last_zscore >= 1.0:
        #if tts_interrupt_score >= 10.0:
            self.tts_interrupted_chunk_input_ids = tts_chunk_input_ids
            #print("TTS stream interrupted")
            return out_chunk_input_ids
        else:
            # if self.tts_interrupted_chunk_input_ids is not None:
            #     print("TTS stream resumed")
            self.tts_interrupted_chunk_input_ids = None
        # substitute the generated audio tokens for the corresponding tts tokens
        start_frame = self.total_frames - len(out_chunk_input_ids) * 2
        self.set_audio_tokens(tts_chunk_input_ids, start_frame=start_frame, channel=0)
        return tts_chunk_input_ids

    def whisper_trans(self) -> Optional[List[int]]:
        if self.resources.whisper_model is None:
            raise ValueError("Whisper model is not loaded.")
        last_trans = self.last_transcription
        trans_audio_start_secs = last_trans["end_secs"] + self.config.chunk_size_secs if last_trans is not None else 0.0
        trans_audio_start_samples = int(trans_audio_start_secs * self.resources.audio_tokenizer.sampling_rate)
        trans_audio_start_chunks, rem_samples = divmod(trans_audio_start_samples, self.chunk_size_samples)
        trans_audio = np.concatenate(self.audio_history_ch2[trans_audio_start_chunks:])[rem_samples:]
        transcription = self._whisper_trans(trans_audio)
        transcription = self._clean_whisper_text(transcription)
        if not transcription:
            return None
        transcription = f" {transcription}"
        trans_input_ids = self.resources.tokenizer.encode(transcription, add_special_tokens=False)
        return trans_input_ids

    def _whisper_trans(self, trans_audio: Union[Tuple[int, np.ndarray], np.ndarray]) -> str:
        # TODO: we assume that whisper uses the same sampling rate as the codec (16kHz), which is correct for now. This could change
        # in the future if a different codec is used. A better approach would be to refactor _prep_audio_for_tokenization out of audio_tokenizer
        # and pass in whisper's sampling rate explicitly.
        trans_audio = self.resources.audio_tokenizer._prep_audio_for_tokenization(trans_audio)
        trans_audio = pad_or_trim(trans_audio, max(trans_audio.shape[-1], int(1.2*self.resources.audio_tokenizer.sampling_rate)), pad_side="left")
        segments = self.resources.whisper_model.transcribe(
            trans_audio, 
            temperature=self.config.trans_temperature,
            language="en",
            no_context=True, 
            single_segment=True, 
            print_progress=False,
        )
        transcription = " ".join([segment.text for segment in segments])
        return transcription
    
    def _clean_whisper_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("[ ", "[")
        text = text.replace(" ]", "]")
        text = text.replace("[blank_audio]", "")
        text = text.replace("[inaudible]", "")
        text = text.replace("[silence]", "")
        text = text.replace("[pause]", "")
        text = text.replace("mm-hmm", "mhm")
        text = text.replace("...", "")
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.replace(">>", "")
        text = text.strip()
        return text

    def measure_event_prob(self):
        logits = np.ctypeslib.as_array(self.resources.llm._ctx.get_logits(), shape=(self.resources.llm._n_vocab,))
        probs = softmax(logits)
        end_audio_prob = probs[self.end_audio_token_id]
        self.stats.event_prob.add_value(end_audio_prob.item())

        if self.stats.event_prob.last_zscore >= 0.0:
            self.prob_event_speaker_token_id = self.get_probable_event_speaker()
        else:
            self.prob_event_speaker_token_id = None
        
    def get_probable_event_speaker(self) -> int:
        _ = next(self.resources.llm.generate([self.end_audio_token_id], reset=False))
        logits = np.ctypeslib.as_array(self.resources.llm._ctx.get_logits(), shape=(self.resources.llm._n_vocab,))
        probs = softmax(logits)
        agent_prob, user_prob = probs[[self.agent_speaker_token_id, self.user_speaker_token_id]]
        event_speaker = self.agent_speaker_token_id if agent_prob > user_prob else self.user_speaker_token_id
        self.resources.llm.n_tokens -= 1
        return event_speaker

    def update_inactivity_timers(self) -> None:
        prev_ch2_abs_max_zscore = self.stats.ch_abs_max.last_zscore[1]
        self.stats.ch_abs_max.add_value((
            np.abs(self.audio_history_ch1[-1]).max().item(), 
            np.abs(self.audio_history_ch2[-1]).max().item(),
        ))
        
        # channel 2 (input)
        if self.stats.ch_abs_max.last_zscore[1] >= 0.0:
            self.ch2_inactivity_elapsed_secs = 0.0
            if prev_ch2_abs_max_zscore < 0.0:
                self.ch2_activity_start_secs = self.total_secs
        else:
            self.ch2_inactivity_elapsed_secs += self.config.chunk_size_secs
        
        # channel 1 (output)
        if self.stats.ch_abs_max.last_zscore[0] >= 0.0:
            self.ch1_inactivity_elapsed_secs = 0.0
        else:
            self.ch1_inactivity_elapsed_secs += self.config.chunk_size_secs

    def should_force_transcription(self) -> bool:
        if self.config.force_trans_after_inactivity_secs == 0.0:
            return False
        return self.ch2_inactivity_elapsed_secs >= self.config.force_trans_after_inactivity_secs \
            and self.stats.event_prob.last_zscore >= 1.0 \
            and self.prob_event_speaker_token_id == self.user_speaker_token_id
    
    def should_force_response(self) -> bool:
        if self.config.force_response_after_inactivity_secs == 0.0:
            return False
        return min(self.ch1_inactivity_elapsed_secs, self.ch2_inactivity_elapsed_secs) >= self.config.force_response_after_inactivity_secs

    def process_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        with self.profilers.total_profiler:
            # Sanity check - input size
            assert audio_chunk.shape[-1] == self.chunk_size_samples, \
                f"audio_chunk must have length {self.chunk_size_samples}, but got {audio_chunk.shape[-1]}"
            tts_chunk_input_ids = None

            # Encode audio chunk to input ids
            with self.profilers.audio_tokenize_profiler:
                audio_chunk_str = self.resources.audio_tokenizer.tokenize_audio(audio_chunk)
            with self.profilers.tokenize_profiler:
                audio_chunk_input_ids = self.resources.tokenizer.encode(audio_chunk_str, add_special_tokens=False)
                if self.config.use_external_tts and self.tts_interrupted_chunk_input_ids is not None:
                    tts_chunk_input_ids = self.tts_interrupted_chunk_input_ids
                elif self.config.use_external_tts:
                    tts_chunk = self.tts_client.next_chunk()
                    if tts_chunk is None and not self.config.external_tts_allow_fallback:
                        tts_chunk = self.default_tts_fallback_chunk
                    if tts_chunk is not None:
                        tts_chunk_input_ids = self.resources.tokenizer.encode(tts_chunk, add_special_tokens=False)
                        # Sanity check - TTS chunk size
                        assert len(tts_chunk_input_ids) == len(audio_chunk_input_ids), \
                            f"TTS chunk must have {len(audio_chunk_input_ids)} tokens, but got {len(tts_chunk_input_ids)}"
            
            # Generate next audio chunk (interleaved by frame with the input audio chunk) and also generate any interleaved text
            with self.profilers.lm_profiler:
                force_trans = self.should_force_transcription()
                force_response = self.should_force_response()
                out_chunk_input_ids = self.process_audio_input_ids(audio_chunk_input_ids, force_trans, force_response)
                out_chunk_input_ids = self.process_tts_input_ids(tts_chunk_input_ids, out_chunk_input_ids)

            # Decode input ids to audio chunk and append to the audio history
            out_chunk = self.detokenize_output_chunk(out_chunk_input_ids)
            self.audio_history_ch2.append(audio_chunk)

            # Update stats and timers
            self.measure_event_prob()
            self.update_inactivity_timers()

            # Sanity check - output size
            assert out_chunk.shape[-1] == self.chunk_size_samples, \
                f"out_chunk must have length {self.chunk_size_samples}, but got {out_chunk.shape[-1]}"
            return out_chunk

    def detokenize_output_chunk(self, out_chunk_input_ids: List[int]) -> np.ndarray:
        with self.profilers.detokenize_profiler:
            out_chunk_str = self.resources.tokenizer.decode(out_chunk_input_ids, skip_special_tokens=False)
        with self.profilers.audio_detokenize_profiler:
            (_, out_chunk), _, preroll_samples = self.resources.audio_tokenizer.detokenize_audio(out_chunk_str, preroll_samples=self.crossfade_ramps[0])
        out_chunk = pad_or_trim(out_chunk, self.chunk_size_samples + preroll_samples)
        if self.config.target_volume_rms > 0:
            out_chunk = normalize_audio_rms(out_chunk, target_rms=self.config.target_volume_rms)
        if len(self.audio_history_ch1) > 0:
            joined_ch1_chunks = smooth_join(self.audio_history_ch1[-1], out_chunk, *self.crossfade_ramps)
            # sanity check - joined size
            assert joined_ch1_chunks.shape[-1] == 2 * self.chunk_size_samples, \
                f"joined_ch1_chunks must have length {2 * self.chunk_size_samples}, but got {joined_ch1_chunks.shape[-1]}"
            self.audio_history_ch1[-1] = joined_ch1_chunks[:self.chunk_size_samples]
            self.audio_history_ch1.append(joined_ch1_chunks[self.chunk_size_samples:])
            # Emit the output chunk from the audio history, shifted left by chunk_fade_secs.
            # This is done because the smooth join (crossfade) modifies the tail of the previous chunk.
            # So, we include the tail of the previous chunk in the output and exclude the tail of the current chunk.
            out_chunk = joined_ch1_chunks[-self.chunk_size_samples-self.crossfade_ramps[0]:-self.crossfade_ramps[0]]
        else:
            self.audio_history_ch1.append(out_chunk)
            # In case of the first chunk there will be no previous chunk tail so pad it on the left with zeros
            out_chunk = pad_or_trim(out_chunk[:-self.crossfade_ramps[0]], self.chunk_size_samples, pad_side="left")
        return out_chunk

    def update_transcript(self, text_start_pos: int, external_pos_ranges: List[Tuple[int, int]] = []) -> None:
        if text_start_pos is None:
            warn(
                "No text start position found, skipping transcript update. "
                "This means that the model did not previously generate a end_audio_token. "
                "This should never happen - investigate this!"
            )
            return
        decode_input_ids = self.input_ids[text_start_pos:-1]
        for start_pos, end_pos in reversed(external_pos_ranges):
            decode_input_ids.insert(end_pos-text_start_pos, self.external_marker_token_id)
            decode_input_ids.insert(start_pos-text_start_pos, self.external_marker_token_id)
        text_str = self.resources.tokenizer.decode(decode_input_ids, skip_special_tokens=False)
        # It is possible for there to be multiple speaker utterances in the same text chunk, so we extract them with regex
        for speaker, sp_text in self.transcript_regex.findall(text_str):
            sp_text_with_external_markers = sp_text.lstrip()
            sp_text = sp_text_with_external_markers.replace(self.config.external_marker_token, "").lstrip()
            if speaker != self.config.agent_identity:
                last_trans = self.last_transcription
                last_trans_end_secs = last_trans["end_secs"] if last_trans is not None else 0.0
                start_secs = max(self.ch2_activity_start_secs, last_trans_end_secs) \
                    if self.ch2_activity_start_secs < self.total_secs-self.ch2_inactivity_elapsed_secs else last_trans_end_secs
                end_secs = self.total_secs
            else:
                start_secs = self.total_secs
                end_secs = None
                if self.config.use_external_tts:
                    self.tts_client.prep_stream(sp_text)
                    self.tts_interrupted_chunk_input_ids = None
            self.transcript.append({
                "speaker": speaker,
                "text": sp_text,
                "start_secs": start_secs,
                "end_secs": end_secs,
                "text_start_pos": text_start_pos, # TODO: this is not accurate if there are multiple utterances
                "text_with_external_markers": sp_text_with_external_markers,
            })
        self.transcript.sort(key=lambda x: x["start_secs"])

    def finalize_last_response(self) -> None:
        last_response = self.last_response
        if last_response is None:
            return
        if last_response.get("planned_text"):
            return
        last_response["planned_text"] = last_response["text"]
        last_response_start_secs = last_response["start_secs"]
        last_response_end_secs = min(self.total_secs - self.ch1_inactivity_elapsed_secs, last_response_start_secs + 10.0)
        if last_response_end_secs <= last_response_start_secs:
            # TODO: no response audio - response should be removed from transcript?
            return
        last_response["end_secs"] = last_response_end_secs
        last_response_audio_input_ids = self.get_audio_tokens(last_response_start_secs, last_response_end_secs)
        c = self.config
        af_ctx_prompt = "".join([
            c.header_audio_first_token,
            c.header_speaker_token,
            f" {c.agent_identity}",
            c.header_speaker_token,
            f" {c.user_identity}",
            c.end_header_token,
        ])
        af_ctx_input_ids = self.resources.tokenizer.encode(af_ctx_prompt)
        af_ctx_input_ids.extend(
            last_response_audio_input_ids + 
            [self.end_audio_token_id, self.agent_speaker_token_id] + 
            self.resources.tokenizer.encode(":", add_special_tokens=False)
        )
        to_ctx_prompt = "".join([
            c.header_text_only_token,
            c.header_speaker_token,
            f" {c.agent_identity}",
            c.header_speaker_token,
            f" {c.user_identity}",
            c.end_header_token,
            f" {c.agent_identity}:",
        ])
        to_ctx_input_ids = self.resources.tokenizer.encode(to_ctx_prompt)
        txt_input_ids = self.resources.tokenizer.encode(" " + last_response["text"], add_special_tokens=False)

        af_probs = np.exp(self.resources.aux_llm.get_logprobs(af_ctx_input_ids, txt_input_ids))
        to_probs = np.exp(self.resources.aux_llm.get_logprobs(to_ctx_input_ids, txt_input_ids))
        probs_ratio = af_probs / to_probs

        tol = 2
        counter = 0
        for i, ratio in enumerate(probs_ratio):
            if ratio >= 1.0:
                counter = 0
            else:
                counter += 1
            if counter > tol:
                i -= counter
                break
        final_text_input_ids = txt_input_ids[:i+1]
        if len(final_text_input_ids) == len(txt_input_ids):
            return
        elif len(final_text_input_ids) == 0:
            final_text_input_ids = self.resources.tokenizer.encode(" [silence]", add_special_tokens=False)
        # else:
        #     final_text_input_ids.append(self.resources.tokenizer.encode("-", add_special_tokens=False)[0])
        last_response["text"] = self.resources.tokenizer.decode(final_text_input_ids, skip_special_tokens=False).lstrip()
        # update sequence
        text_start_pos = last_response["text_start_pos"] + 2
        text_end_pos = text_start_pos + len(txt_input_ids)
        prev_input_ids_len = len(self.input_ids)
        self.input_ids = self.input_ids[:text_start_pos] + final_text_input_ids + self.input_ids[text_end_pos:]
        self.recompute_kv_cache(text_start_pos, text_end_pos)
        input_ids_diff = len(self.input_ids) - prev_input_ids_len
        # update self.audio_tokens_idx
        if input_ids_diff != 0:
            for i in range(self.total_frames-1, -1, -1):
                if self.audio_tokens_idx[i] <= text_end_pos:
                    break
                self.audio_tokens_idx[i] += input_ids_diff

    def frames_from_secs(self, secs: float) -> int:
        frames = int(secs * self.resources.audio_tokenizer.framerate * 2)
        # make sure we return the position at the beginning of an audio token pair
        if frames % 2 != 0:
            frames -= 1
        return frames

    def get_audio_tokens(self, start_secs: Optional[float] = None, end_secs: Optional[float] = None) -> List[int]:
        start_frame = 0 if start_secs is None else self.frames_from_secs(start_secs)
        end_frame = self.total_frames if end_secs is None else self.frames_from_secs(end_secs)
        audio_tokens = [self.input_ids[i] for i in self.audio_tokens_idx[start_frame:end_frame]]
        return audio_tokens
    
    def set_audio_tokens(
        self, 
        audio_tokens: List[int], 
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None, 
        channel: Optional[int] = None, 
    ) -> None:
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.total_frames if end_frame is None else end_frame
        audio_tokens_idx = self.audio_tokens_idx[start_frame:end_frame]
        if channel is not None:
            audio_tokens_idx = audio_tokens_idx[channel::2]
        assert len(audio_tokens_idx) == len(audio_tokens), \
            f"({len(audio_tokens)}) were provided, but ({len(audio_tokens_idx)}) exist in range [{start_frame}, {end_frame}) on channel {channel}."
        for token_idx, new_token in zip(audio_tokens_idx, audio_tokens):
            self.input_ids[token_idx] = new_token
        self.recompute_kv_cache(audio_tokens_idx[0], audio_tokens_idx[-1]+1)
        
    def recompute_kv_cache(self, edit_start_pos: int, edit_end_pos: Optional[int] = None) -> None:
        trim_to_frames = self.frames_from_secs(self.trim_to_secs)
        trim_to_pos = self.audio_tokens_idx[trim_to_frames]
        if trim_to_frames == 0 or edit_end_pos is None or edit_end_pos > trim_to_pos:
            start_pos = edit_start_pos if trim_to_frames == 0 else max(edit_start_pos, trim_to_pos)
            self.resources.llm.n_tokens = start_pos if trim_to_frames == 0 else start_pos-trim_to_pos+self.context_start_pos
            audio_mode = all([t > self.end_header_token_id for t in self.input_ids[-2:]])
            last_n = 2 if audio_mode else 1
            self.resources.llm.eval(self.input_ids[start_pos:-last_n])

    def get_sequence_str(self) -> str:
        agent_sequence = self.resources.tokenizer.decode(self.input_ids, skip_special_tokens=False)
        return agent_sequence
    
    def get_audio_history(self) -> np.ndarray:
        if len(self.audio_history_ch1) == 0:
            return np.zeros((2, 0), dtype=np.float32)
        audio_history = np.stack(
            [
                np.concatenate(self.audio_history_ch1), 
                np.concatenate(self.audio_history_ch2),
            ], 
            axis=0,
        )
        return audio_history
    
    def _format_time_for_transcript(self, secs: float) -> str:
        hours, rem = divmod(secs, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours)}:{int(minutes):02}:{seconds:06.3f}"
    
    def _format_start_end_for_transcript(self, entry: Dict[str, Any]) -> str:
        start_str = self._format_time_for_transcript(entry["start_secs"])
        end_str = self._format_time_for_transcript(entry["end_secs"] if entry["end_secs"] is not None else self.total_secs)
        return f"{start_str} - {end_str}"

    def format_transcript(self) -> str:
        formatted_lines = []
        for entry in self.transcript:
            if "planned_text" in entry and entry["text"] != entry["planned_text"]:
                #TODO: handle external markers in planned_text
                planned_text = entry["planned_text"] \
                    if entry["text"] == "[silence]" else entry["planned_text"][len(entry["text"]):].lstrip()
                entry_text = f"{entry['text']}  ⟶  {{{planned_text}}}"
            else:
                entry_text = entry["text_with_external_markers"]
            formatted_lines.append(f"[{self._format_start_end_for_transcript(entry)}] {entry["speaker"]}: {entry_text}")
        formatted_transcript = "\n".join(formatted_lines)
        return formatted_transcript
    
    def get_external_llm_messages(self) -> List[Dict[str, str]]:
        if self.llm_client is None:
            return None
        return self.llm_client.get_messages(self.transcript, self.config.external_llm_instructions)

@dataclass
class RealtimeAgentMultiprocessingInfo:
    config: RealtimeAgentConfig
    sampling_rate: int
    chunk_size_samples: int
    total_secs: float
    transcript: str
    sequence: str
    audio_history: np.ndarray
    external_llm_messages: Optional[List[Dict[str, str]]]

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
                        total_secs=agent.total_secs,
                        transcript=agent.format_transcript(),
                        sequence=agent.get_sequence_str(),
                        audio_history=agent.get_audio_history(),
                        external_llm_messages=agent.get_external_llm_messages(),
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

    def next_output(self, block: bool = False):
        if not block and self.output_queue.empty():
            return None
        return self.output_queue.get()
    