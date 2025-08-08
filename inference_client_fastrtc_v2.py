from fastrtc import StreamHandler, Stream, AdditionalOutputs
from queue import Queue, Empty
from datetime import datetime
import gradio as gr
import numpy as np
import soundfile as sf
import argparse

from realtime_codec_agent.realtime_agent_v2 import RealtimeAgent, RealtimeAgentResources

class AgentHandler(StreamHandler):
    def __init__(self, agent: RealtimeAgent, report_interval_secs: float = 1.0):
        super().__init__(
            input_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
            output_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
        )
        self.agent = agent
        self.report_interval_secs = report_interval_secs

        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.queue = Queue()
        self.started = False
        self.realtime_factor_sum = 0.0
        self.report_chunk_count = 0
        self.last_report_time = datetime.now()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.started:
            return
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
        
        start_time = datetime.now()
        
        # Suppress low amplitude noise from the microphone
        if np.abs(chunk).max() < 100:
            chunk = np.zeros_like(chunk)

        print(f'Data from microphone:{chunk.shape, chunk.dtype, chunk.min(), chunk.max()}')

        # Process the chunk and get the next output
        chunk = chunk.squeeze(0).astype(np.float32) / 32768.0
        out_chunk = self.agent.process_audio(chunk)
        out_chunk = np.expand_dims((out_chunk * 32767.0).astype(np.int16), axis=0)
        
        print(f'Data from model:{out_chunk.shape, out_chunk.dtype, out_chunk.min(), out_chunk.max()}')

        # Compute info for reporting
        end_time = datetime.now()
        elapsed_secs = (end_time - start_time).total_seconds()
        self.realtime_factor_sum += self.agent.config.chunk_size_secs / elapsed_secs
        self.report_chunk_count += 1

        # Report if enough time has passed
        if (end_time - self.last_report_time).total_seconds() >= self.report_interval_secs:
            realtime_factor = self.realtime_factor_sum / self.report_chunk_count
            self.realtime_factor_sum = 0.0
            self.report_chunk_count = 0
            self.last_report_time = end_time
            return (self.output_sample_rate, out_chunk), AdditionalOutputs(f"{realtime_factor:.2f}x")
        
        return (self.output_sample_rate, out_chunk)

    def copy(self):
        return AgentHandler(self.agent, self.report_interval_secs)

    def shutdown(self):
        if not self.started:
            print(">>> Not Started <<<")
            return
        transcript = self.agent.format_transcript()
        sequence = self.agent.get_sequence_str()
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- Transcript:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(transcript)
            f.write("\n\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- Sequence:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(sequence)
            f.write("\n")
        audio_history = self.agent.get_audio_history()
        audio_history = (audio_history * 32767.0).astype(np.int16)
        sf.write("output.wav", audio_history.T, self.output_sample_rate)
        
        self.started = False
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        self.set_config_and_reset()
        self.started = True
        print(">>> Started <<<")

    def set_config_and_reset(self):
        if not self.phone_mode:
            self.wait_for_args_sync()
            config = self.agent.config
            config.agent_opening_text = self.latest_args[1]
            config.agent_voice_enrollment = self.latest_args[2]
            config.chunk_size_secs = float(self.latest_args[3])
            config.temperature = float(self.latest_args[4])
            config.top_k = int(self.latest_args[5])
            config.top_p = float(self.latest_args[6])
            config.min_p = float(self.latest_args[7])
            config.presence_penalty = float(self.latest_args[8])
            config.frequency_penalty = float(self.latest_args[9])
            config.max_context_secs = float(self.latest_args[10])
            config.trim_by_secs = float(self.latest_args[11])

            if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
                config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

            self.agent.set_config(config)
        self.agent.reset()

def display_handler(component1, realtime_factor: str):
    return realtime_factor

def main(args):
    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            llm_model_path=args.llm_model_path,
        ),
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
        additional_inputs=[
            gr.Textbox("hello how are you?", label="Agent Opening Text"),
            gr.Audio(label="Agent Voice Enrollment"),
            gr.Slider(0.02, 1.0, value=0.1, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature"),
            gr.Slider(0, 500, value=100, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Frequency Penalty"),
            gr.Number(80.0, minimum=5.0, maximum=80.0, step=5.0, label="Max Context Length (seconds)"),
            gr.Number(20.0, minimum=1.0, maximum=20.0, step=1.0, label="Trim By (seconds)"),
        ],
        additional_outputs=[
            gr.Textbox(label="Realtime Factor"),
        ],
        additional_outputs_handler=display_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Realtime Codec Agent with FastRTC.")
    parser.add_argument(
        "--llm_model_path", 
        default="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-2-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-2-BF16.gguf", 
        help="Path to the model GGUF file.",
    )

    args = parser.parse_args()

    main(args)