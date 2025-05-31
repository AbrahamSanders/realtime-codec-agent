from fastrtc import StreamHandler, Stream
from queue import Queue, Empty
import numpy as np
import soundfile as sf

from realtime_codec_agent.realtime_agent import RealtimeAgent, RealtimeAgentResources, RealtimeAgentConfig

class AgentHandler(StreamHandler):
    def __init__(self, agent: RealtimeAgent) -> None:
        super().__init__(
            input_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
            output_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
        )
        self.agent = agent
        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.queue = Queue()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, frame_audio = frame
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= self.agent.chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [self.agent.chunk_size_samples], axis=1)
            self.queue.put(chunk)

    def emit(self) -> None:
        try:
            chunk = self.queue.get_nowait()
        except Empty:
            return None
        
        print(f'Data from microphone:{chunk.shape, chunk.dtype, chunk.min(), chunk.max()}')
        
        # Suppress low amplitude noise from the microphone
        if np.abs(chunk).max() < 100:
            chunk = np.zeros_like(chunk)
        
        # Process the chunk and get the next output
        chunk = chunk.squeeze(0).astype(np.float32) / 32768.0
        out_chunk = self.agent.process_audio(chunk)
        out_chunk = np.expand_dims((out_chunk * 32767.0).astype(np.int16), axis=0)
        
        print(f'Data from model:{out_chunk.shape, out_chunk.dtype, out_chunk.min(), out_chunk.max()}')
        return (self.output_sample_rate, out_chunk)

    def copy(self):
        return AgentHandler(self.agent)

    def shutdown(self):
        audio_first_sequence, text_first_sequence = self.agent.get_sequence_strs()
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- Audio First Sequence:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(audio_first_sequence)
            f.write("\n\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- Text First Sequence:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(text_first_sequence)
            f.write("\n")
        audio_history = self.agent.get_audio_history()
        audio_history = (audio_history * 32767.0).astype(np.int16)
        sf.write("output.wav", audio_history.T, self.output_sample_rate)
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        self.agent.reset()
        print(">>> Started <<<")

def main():
    agent = RealtimeAgent(
        resources=RealtimeAgentResources(mock=False),
        config=RealtimeAgentConfig(chunk_size_secs=0.7),
    )
    handler = AgentHandler(agent)
    stream = Stream(
        handler=handler,
        track_constraints = {
            "echoCancellation": False,
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            "sampleRate": {"ideal": handler.input_sample_rate},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        },
        modality="audio", 
        mode="send-receive",
    )
    stream.ui.launch()

if __name__ == "__main__":
    main()