import numpy as np
from typing import List

from .realtime_agent_resources import RealtimeAgentResources
from .realtime_agent_config import RealtimeAgentConfig
from .realtime_agent_v2 import RealtimeAgent

class RealtimeAgentSelfPlay:
    def __init__(self, resources: RealtimeAgentResources = None, config_1: RealtimeAgentConfig = None, config_2: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        resources_2 = resources.clone_for_self_play()

        self.agent_1 = RealtimeAgent(resources, config_1)
        self.agent_2 = RealtimeAgent(resources_2, config_2)

        self.reset()

    def reset(self) -> None:
        for agent in [self.agent_1, self.agent_2]:
            agent.reset()

    def set_config(self, config_1: RealtimeAgentConfig, config_2: RealtimeAgentConfig) -> None:
        for agent, config in [(self.agent_1, config_1), (self.agent_2, config_2)]:
            agent.set_config(config)

    def next_chunk_input_ids(self) -> List[int]:
        force_trans_1 = self.agent_1.should_force_transcription()
        force_response_1 = self.agent_1.should_force_response()
        force_trans_2 = self.agent_2.should_force_transcription()
        force_response_2 = self.agent_2.should_force_response()

        # Add the first output token on agent_2's sequence to offset it by one, since it will be acting as agent_1's "B" channel
        if self.agent_2.total_frames == 0:
            # Generate an initial "dummy" token
            self.agent_2.process_audio_input_ids()
        else:
            # Use the last agent_2 output token from the previous chunk, which was removed from agent_2's input_ids
            # to avoid messing up timing calculations
            self.agent_2.input_ids.append(self.agent_1.input_ids[-1])
            self.agent_2.audio_tokens_idx.append(len(self.agent_2.input_ids)-1)
            self.agent_2.resources.llm.eval(self.agent_2.input_ids[-1:])

        out_chunk_input_ids_ch1 = [0] * self.agent_1.chunk_size_frames_per_channel
        out_chunk_input_ids_ch2 = [0] * self.agent_1.chunk_size_frames_per_channel
        for i in range(len(out_chunk_input_ids_ch1)):
            # Generate agent 1's next output token
            next_token_1 = self.agent_1.process_audio_input_ids(None, force_trans_1, force_response_1)[0]
            out_chunk_input_ids_ch1[i] = next_token_1
            force_trans_1 = force_response_1 = False
            # Feed agent 1's output token into agent 2 as the next "input" token
            self.agent_2.input_ids.append(next_token_1)
            self.agent_2.audio_tokens_idx.append(len(self.agent_2.input_ids)-1)

            # Generate agent 2's next output token
            next_token_2 = self.agent_2.process_audio_input_ids(None, force_trans_2, force_response_2)[0]
            out_chunk_input_ids_ch2[i] = next_token_2
            force_trans_2 = force_response_2 = False
            # Feed agent 2's output token into agent 1 as the next "input" token
            self.agent_1.input_ids.append(next_token_2)
            self.agent_1.audio_tokens_idx.append(len(self.agent_1.input_ids)-1)

        # remove the last output token from agent_2's input_ids to keep the timing calculations correct
        self.agent_2.input_ids.pop()
        self.agent_2.audio_tokens_idx.pop()
        self.agent_2.resources.llm.n_tokens -= 1
        return out_chunk_input_ids_ch1, out_chunk_input_ids_ch2

    def next_chunk(self) -> np.ndarray:
        with self.agent_1.profilers.total_profiler, self.agent_2.profilers.total_profiler:
            with self.agent_1.profilers.lm_profiler, self.agent_2.profilers.lm_profiler:
                out_chunk_input_ids_ch1, out_chunk_input_ids_ch2 = self.next_chunk_input_ids()
            out_chunk = []
            for agent, out_chunk_input_ids in [(self.agent_1, out_chunk_input_ids_ch1), (self.agent_2, out_chunk_input_ids_ch2)]:
                out_chunk_ch = agent.detokenize_output_chunk(out_chunk_input_ids)
                out_chunk.append(out_chunk_ch)
            out_chunk = np.array(out_chunk)

            # Copy latest ch1 chunk into ch2 of the opposite agent's audio history
            for agent_a, agent_b in [(self.agent_1, self.agent_2), (self.agent_2, self.agent_1)]:
                agent_a.audio_history_ch2.append(agent_b.audio_history_ch1[-1])
                # the second to last ch1 chunk is also updated during the smooth join so needs to be copied again
                if len(agent_a.audio_history_ch2) > 1:
                    agent_a.audio_history_ch2[-2] = agent_b.audio_history_ch1[-2]

            # Update stats and timers
            for agent in [self.agent_1, self.agent_2]:
                agent.measure_event_prob()
                agent.update_inactivity_timers()

            # Sanity check - output size
            assert out_chunk.shape[-1] == self.agent_1.chunk_size_samples, \
                f"out_chunk must have length {self.agent_1.chunk_size_samples}, but got {out_chunk.shape[-1]}"
            return out_chunk
