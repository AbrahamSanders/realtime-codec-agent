import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RealtimeAgentConfig:
    agent_opening_text: str = None
    agent_voice_enrollment: Tuple[int, np.ndarray] = None
    agent_identity: str = "A"
    user_identity: str = "B"
    temperature: float = 1.0
    trans_temperature: float = 0.0
    force_trans_after_activity: bool = True
    force_trans_margin_secs: float = 0.5
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
    seed: Optional[int] = None
    header_agent_token: str = "<|agent|>"
    header_agent_voice_token: str = "<|agent_voice|>"
    header_speaker_token: str = "<|speaker|>"
    end_header_token: str = "<|end_header|>"
    start_audio_token: str = "<|audio|>"
    end_audio_token: str = "<|end_audio|>"
    run_profilers: bool = False
    profiler_report_interval_secs: float = 2.0

    def __post_init__(self):
        if int(self.chunk_size_secs*100) % 2 != 0:
            raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
        if self.chunk_fade_secs > self.chunk_size_secs:
            raise ValueError("Chunk fade length cannot be longer than the chunk size.")
