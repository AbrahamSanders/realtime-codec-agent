import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RealtimeAgentConfig:
    agent_opening_text: Optional[str] = "hello?"
    agent_voice_enrollment: Optional[Tuple[int, np.ndarray]] = None
    agent_identity: str = "A"
    user_identity: str = "B"
    temperature: float = 1.0
    trans_temperature: float = 0.0
    force_trans_after_inactivity_secs: float = 1.0
    use_whisper: bool = True
    top_k: int = 100
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    chunk_size_secs: float = 0.1
    chunk_fade_secs: float = 0.02
    max_context_secs: float = 80.0
    trim_by_secs: float = 20.0
    target_volume_rms: float = 0.0
    force_response_after_inactivity_secs: float = 3.0
    activity_abs_max_threshold: float = 100 / 32768.0
    seed: Optional[int] = 42
    header_audio_first_token: str = "<|audio_first|>"
    header_text_only_token: str = "<|text_only|>"
    header_agent_token: str = "<|agent|>"
    header_agent_voice_token: str = "<|agent_voice|>"
    header_speaker_token: str = "<|speaker|>"
    end_header_token: str = "<|end_header|>"
    start_audio_token: str = "<|audio|>"
    end_audio_token: str = "<|end_audio|>"
    use_external_llm: bool = False
    external_llm_api_key: Optional[str] = None
    external_llm_base_url: Optional[str] = None
    external_llm_model: str = "gpt-4o-mini"
    external_llm_top_p: float = 0.9
    external_llm_instructions: Optional[str] = None
    external_llm_suppress_threshold: float = 0.1
    use_external_tts: bool = False
    external_tts_server_url: str = "http://127.0.0.1:8001"
    external_tts_prompt_text: Optional[str] = None
    run_profilers: bool = True
    profiler_report_interval_secs: float = 2.0

    def __post_init__(self):
        if int(self.chunk_size_secs*100) % 2 != 0:
            raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
        if self.chunk_fade_secs > self.chunk_size_secs:
            raise ValueError("Chunk fade length cannot be longer than the chunk size.")
