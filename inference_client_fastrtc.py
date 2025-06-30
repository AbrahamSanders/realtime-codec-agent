from typing import Optional
from fastrtc import StreamHandler, Stream, AdditionalOutputs
from queue import Queue, Empty
from warnings import warn
from datetime import datetime
import gradio as gr
import numpy as np
import soundfile as sf
import argparse

from realtime_codec_agent.realtime_agent import RealtimeAgent, RealtimeAgentResources, RealtimeAgentConfig
from realtime_codec_agent.asr_handler import ASRHandlerMultiprocessing, ASRConfig

class AgentHandler(StreamHandler):
    def __init__(self, agent: RealtimeAgent, asr_handler: Optional[ASRHandlerMultiprocessing] = None, report_interval_secs: float = 1.0):
        super().__init__(
            input_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
            output_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
        )
        self.agent = agent
        self.asr_handler = asr_handler
        self.report_interval_secs = report_interval_secs
        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.queue = Queue()
        self.started = False
        self.realtime_factor_sum = 0.0
        self.report_chunk_count = 0
        self.last_report_time = datetime.now()

        if self.asr_handler is None and not self.agent.config.enable_audio_first_transcription:
            warn(
                "Audio-first transcription is disabled and no asr_handler was provided. "
                "The agent will not receive input audio transcription and will probably not function correctly.",
            )

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.started:
            return
        _, frame_audio = frame
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= self.agent.chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [self.agent.chunk_size_samples], axis=1)
            if self.asr_handler is not None:
                self.asr_handler.queue_input((self.input_sample_rate, chunk.squeeze(0)))
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
        
        # Get the next transcription chunk from the ASR handler
        trans_chunk = None
        if self.asr_handler is not None:
            trans_chunk = self.asr_handler.next_output()

        # Process the chunk and get the next output
        chunk = chunk.squeeze(0).astype(np.float32) / 32768.0
        out_chunk = self.agent.process_audio(chunk, trans_chunk)
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
        return AgentHandler(self.agent, self.asr_handler)

    def shutdown(self):
        if not self.started:
            print(">>> Not Started <<<")
            return
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
            config.agent_voice_enrollment_text = self.latest_args[3]
            config.chunk_size_secs = float(self.latest_args[4])
            config.text_first_temperature = float(self.latest_args[5])
            config.text_first_top_p = float(self.latest_args[6])
            config.text_first_min_p = float(self.latest_args[7])
            config.text_first_presence_penalty = float(self.latest_args[8])
            config.text_first_frequency_penalty = float(self.latest_args[9])
            config.audio_first_cont_temperature = float(self.latest_args[10])
            config.audio_first_trans_temperature = float(self.latest_args[11])
            config.max_seq_length = int(self.latest_args[12])
            config.trim_by = int(self.latest_args[13])

            if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
                config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

            self.agent.set_config(config)
        self.agent.reset()

def display_handler(component1, realtime_factor: str):
    return realtime_factor

def main(args):
    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            tensor_parallel_size = args.tensor_parallel_size,
            debug = args.debug,
            mock = args.mock,
        ),
        config=RealtimeAgentConfig(
            enable_audio_first_transcription = not args.whisper,
        ),
    )
    asr_handler = None
    if not agent.config.enable_audio_first_transcription:
        asr_handler = ASRHandlerMultiprocessing(
            config = ASRConfig(
                model_size="small.en",
                n_context_segs=2,
                n_prefix_segs=2,
                max_buffer_size=10,
            ),
            device="cuda:2"
        )
        
    handler = AgentHandler(agent, asr_handler)
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
            gr.Textbox("hello my name is alex, how can i help you?", label="Agent Opening Text"),
            gr.Audio(label="Agent Voice Enrollment"),
            gr.Textbox(label="Agent Voice Enrollment Text"),
            gr.Slider(0.02, 1.0, value=0.1, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Text-First Temperature"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Text-First Top-p"),
            gr.Slider(0.0, 1.0, value=0.002, step=0.001, label="Text-First Min-p"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Text-First Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Text-First Frequency Penalty"),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Audio-First Continuation Temperature"),
            gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Audio-First Transcription Temperature"),
            gr.Number(4096, minimum=512, maximum=8192, precision=0, label="Max Sequence Length (tokens)"),
            gr.Number(1024, minimum=128, maximum=2048, precision=0, label="Trim By (tokens)"),
        ],
        additional_outputs=[
            gr.Textbox(label="Realtime Factor"),
        ],
        additional_outputs_handler=display_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Realtime Codec Agent with FastRTC.")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Sets tensor_parallel_size for vLLM (number of GPUs).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (starts vLLM in eager mode).")
    parser.add_argument("--mock", action="store_true", help="Enable mock mode (does not start vLLM, echos input audio).")
    parser.add_argument("--whisper", action="store_true", help="Use Whisper for ASR instead of native audio-first transcription.")
    args = parser.parse_args()
    main(args)