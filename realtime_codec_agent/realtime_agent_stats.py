from typing import Deque, Optional
from collections import deque
import numpy as np

from .realtime_agent_config import RealtimeAgentConfig

class RealtimeAgentStats:
    def __init__(self, config: RealtimeAgentConfig, window_secs: float = 20.0, update_interval_secs: float = 5.0):
        self.window_chunks = int(window_secs / config.chunk_size_secs)
        self.update_interval_chunks = int(update_interval_secs / config.chunk_size_secs)
        self.reset()
    
    @property
    def last_zscore(self) -> float:
        if not self.values:
            return 0.0
        return self.values_zscores[-1]

    def reset(self):
        self.values: Deque[float] = deque()
        self.values_zscores: Deque[float] = deque()
        self.mean = 0.0
        self.std = 1.0

    def add_value(self, value: float):
        self.values.append(value)
        self.values_zscores.append((value - self.mean) / self.std)
        if len(self.values) > self.window_chunks:
            self.values.popleft()
            self.values_zscores.popleft()
        if len(self.values) < self.update_interval_chunks or len(self.values) % self.update_interval_chunks == 0:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values) if len(self.values) > 1 else 1.0

class RealtimeAgentStatsCollection:
    def __init__(self, config: RealtimeAgentConfig):
        self.ch1_abs_max = RealtimeAgentStats(config)
        self.ch2_abs_max = RealtimeAgentStats(config)
        self.transcription_prob = RealtimeAgentStats(config)
        self.response_prob = RealtimeAgentStats(config)
        self.tts_interrupt_score = RealtimeAgentStats(config)

    def reset(self):
        self.ch1_abs_max.reset()
        self.ch2_abs_max.reset()
        self.transcription_prob.reset()
        self.response_prob.reset()
        self.tts_interrupt_score.reset()
