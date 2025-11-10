import numpy as np
from datetime import datetime
from typing import List, Tuple

from .realtime_agent_config import RealtimeAgentConfig

class RealtimeAgentProfiler:
    def __init__(self, config: RealtimeAgentConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.report_chunk_count: int = 0
        self.realtime_factor_sum: float = 0.0
        self.realtime_factor_values: List[float] = []
        self.chunk_start: datetime = None

    def log_chunk_start(self):
        if not self.config.run_profilers:
            return
        self.chunk_start = datetime.now()

    def log_chunk_end(self):
        if not self.config.run_profilers:
            return
        if self.chunk_start is None:
            raise ValueError("Chunk start time not set. Call log_chunk_start() before log_chunk_end().")
        chunk_end = datetime.now()
        elapsed_secs = (chunk_end - self.chunk_start).total_seconds()
        self.realtime_factor_sum += self.config.chunk_size_secs / (elapsed_secs + 1e-8)
        self.report_chunk_count += 1
        self.chunk_start = None

        if self.report_chunk_count * self.config.chunk_size_secs >= self.config.profiler_report_interval_secs:
            realtime_factor = self.realtime_factor_sum / self.report_chunk_count
            self.realtime_factor_values.append(realtime_factor)
            self.realtime_factor_sum = 0.0
            self.report_chunk_count = 0

    def __enter__(self):
        self.log_chunk_start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.log_chunk_end()

class RealtimeAgentProfilerCollection:
    def __init__(self, config: RealtimeAgentConfig):
        self.config = config
        self.total_profiler = RealtimeAgentProfiler(config)
        self.tokenize_profiler = RealtimeAgentProfiler(config)
        self.detokenize_profiler = RealtimeAgentProfiler(config)
        self.audio_tokenize_profiler = RealtimeAgentProfiler(config)
        self.audio_detokenize_profiler = RealtimeAgentProfiler(config)
        self.lm_profiler = RealtimeAgentProfiler(config)

    def reset(self):
        self.total_profiler.reset()
        self.tokenize_profiler.reset()
        self.detokenize_profiler.reset()
        self.audio_tokenize_profiler.reset()
        self.audio_detokenize_profiler.reset()
        self.lm_profiler.reset()

    def build_plot(self, ylim: Tuple[float, float] = (0.5, 3.0)):
        import matplotlib.pyplot as plt
        x = np.arange(
            self.config.profiler_report_interval_secs, 
            self.config.profiler_report_interval_secs * (len(self.total_profiler.realtime_factor_values) + 1), 
            self.config.profiler_report_interval_secs,
        )
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, self.total_profiler.realtime_factor_values, label="total", color="C0")
        ax.axhline(
            y=np.median(self.total_profiler.realtime_factor_values), 
            xmin=0.05, xmax=0.95, color='C0', linestyle='--', linewidth=1.5, label='total (median)'
        )
        if self.tokenize_profiler.realtime_factor_values:
            ax.plot(x, self.tokenize_profiler.realtime_factor_values, label="tokenize", color="C1")
            ax.axhline(
                y=np.median(self.tokenize_profiler.realtime_factor_values), 
                xmin=0.05, xmax=0.95, color='C1', linestyle='--', linewidth=1.5, label='tokenize (median)'
            )
        if self.detokenize_profiler.realtime_factor_values:
            ax.plot(x, self.detokenize_profiler.realtime_factor_values, label="detokenize", color="C2")
            ax.axhline(
                y=np.median(self.detokenize_profiler.realtime_factor_values), 
                xmin=0.05, xmax=0.95, color='C2', linestyle='--', linewidth=1.5, label='detokenize (median)'
            )
        if self.audio_tokenize_profiler.realtime_factor_values:
            ax.plot(x, self.audio_tokenize_profiler.realtime_factor_values, label="audio_tokenize", color="C3")
            ax.axhline(
                y=np.median(self.audio_tokenize_profiler.realtime_factor_values), 
                xmin=0.05, xmax=0.95, color='C3', linestyle='--', linewidth=1.5, label='audio_tokenize (median)'
            )
        if self.audio_detokenize_profiler.realtime_factor_values:
            ax.plot(x, self.audio_detokenize_profiler.realtime_factor_values, label="audio_detokenize", color="C4")
            ax.axhline(
                y=np.median(self.audio_detokenize_profiler.realtime_factor_values), 
                xmin=0.05, xmax=0.95, color='C4', linestyle='--', linewidth=1.5, label='audio_detokenize (median)'
            )
        if self.lm_profiler.realtime_factor_values:
            ax.plot(x, self.lm_profiler.realtime_factor_values, label="lm", color="C5")
            ax.axhline(
                y=np.median(self.lm_profiler.realtime_factor_values), 
                xmin=0.05, xmax=0.95, color='C5', linestyle='--', linewidth=1.5, label='lm (median)'
            )
        ax.axhline(y=1.0, xmin=0.05, xmax=0.95, color='orange', linestyle='--', linewidth=2.5, label='threshold')
        ax.set_title("Realtime Factor Profile")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Realtime factor")
        ax.set_ylim(*ylim)
        ax.grid(True)
        fig.legend(loc='outside center right')
        return fig