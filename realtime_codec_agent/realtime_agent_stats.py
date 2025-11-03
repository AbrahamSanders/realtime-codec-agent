from typing import Deque, Tuple, Union
from collections import deque
import numpy as np

from .realtime_agent_config import RealtimeAgentConfig

class RealtimeAgentStats:
    def __init__(self, config: RealtimeAgentConfig, value_size: int = 1, window_secs: float = 20.0, update_interval_secs: float = 5.0):
        self.value_size = value_size
        self.window_chunks = int(window_secs / config.chunk_size_secs)
        self.update_interval_chunks = int(update_interval_secs / config.chunk_size_secs)
        self.reset()
    
    @property
    def last_zscore(self) -> Union[float, Tuple[float, ...]]:
        if not self.values:
            return tuple(0.0 for _ in range(self.value_size)) if self.value_size > 1 else 0.0
        return self.values_zscores[-1] if self.value_size > 1 else self.values_zscores[-1][0]

    def reset(self):
        self.values: Deque[Tuple[float, ...]] = deque()
        self.values_zscores: Deque[Tuple[float, ...]] = deque()
        self.mean = 0.0
        self.std = 1.0

    def add_value(self, value: Union[float, Tuple[float, ...]]):
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, float) or isinstance(value, int):
            value = (value,)
        self.values.append(value)
        self.values_zscores.append(tuple((v - self.mean) / self.std for v in value))
        if len(self.values) > self.window_chunks:
            self.values.popleft()
            self.values_zscores.popleft()
        if len(self.values) < self.update_interval_chunks or len(self.values) % self.update_interval_chunks == 0:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values, mean=self.mean) if len(self.values) > 1 else 1.0

class RealtimeAgentStatsCollection:
    def __init__(self, config: RealtimeAgentConfig):
        self.ch_abs_max = RealtimeAgentStats(config, value_size=2)
        self.event_prob = RealtimeAgentStats(config)
        self.tts_interrupt_score = RealtimeAgentStats(config)

    def reset(self):
        self.ch_abs_max.reset()
        self.event_prob.reset()
        self.tts_interrupt_score.reset()
