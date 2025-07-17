from fastrtc import StreamHandler, Stream, AdditionalOutputs
from queue import Queue, Empty
from datetime import datetime
from dotenv import load_dotenv
from websockets.sync.client import connect
from collections import deque
import gradio as gr
import numpy as np
import soundfile as sf
import argparse
import msgpack
import librosa

from realtime_codec_agent.realtime_agent import RealtimeAgent, RealtimeAgentResources, AUDIO_FIRST_TRANSCRIPTION_MODES

class AgentHandler(StreamHandler):
    def __init__(self, agent: RealtimeAgent, kyutai_stt_ws_url: str, kyutai_stt_api_key: str, report_interval_secs: float = 1.0):
        super().__init__(
            input_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
            output_sample_rate=agent.resources.audio_tokenizer.sampling_rate,
        )
        self.agent = agent
        self.kyutai_stt_ws_url = kyutai_stt_ws_url
        self.kyutai_stt_api_key = kyutai_stt_api_key
        self.report_interval_secs = report_interval_secs

        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.queue = Queue()
        self.started = False
        self.realtime_factor_sum = 0.0
        self.report_chunk_count = 0
        self.last_report_time = datetime.now()

        self.stt_websocket = None
        self.stt_in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.stt_chunk_size_samples = int(self.input_sample_rate * 0.08)
        self.stt_sample_rate = 24000
        self.stt_chunk = ""
        self.stt_chunk_start = None
        self.stt_ready_chunks = deque()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.started:
            return
        _, frame_audio = frame
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= self.agent.chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [self.agent.chunk_size_samples], axis=1)
            self.queue.put(chunk)
        self.queue_frame_for_stt(frame_audio)

    def queue_frame_for_stt(self, frame_audio: np.ndarray) -> None:
        if self.stt_websocket is None:
            return
        self.stt_in_buffer = np.concatenate((self.stt_in_buffer, frame_audio), axis=1)
        while self.stt_in_buffer.shape[-1] >= self.stt_chunk_size_samples:
            chunk, self.stt_in_buffer = np.split(self.stt_in_buffer, [self.stt_chunk_size_samples], axis=1)
            chunk = chunk.squeeze(0).astype(np.float32) / 32768.0
            chunk = librosa.resample(chunk, orig_sr=self.input_sample_rate, target_sr=self.stt_sample_rate)
            chunk = {"type": "Audio", "pcm": chunk.tolist()}
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            self.stt_websocket.send(msg)

    def read_stt_result(self) -> None:
        if self.stt_websocket is None:
            return
        messages = []
        while True:
            try:
                messages.append(self.stt_websocket.recv(timeout=0))
            except TimeoutError:
                break
        
        for message in messages:
            data = msgpack.unpackb(message, raw=False)
            
            text = ""
            pause = False
            #pause = data["type"] == "Step" and data["prs"][2] > 0.99 and self.stt_chunk_start is not None
            if data["type"] == "Word":
                text = data["text"]
                self.stt_chunk += f" {text}"
                if self.stt_chunk_start is None:
                    self.stt_chunk_start = data["start_time"]

            if pause or text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith(","):
                self.stt_ready_chunks.append((self.stt_chunk.lstrip(), self.stt_chunk_start))
                self.stt_chunk = ""
                self.stt_chunk_start = None

    def emit(self) -> None:
        # Get the next transcription chunk from the websocket if available
        self.read_stt_result()
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
        next_stt_chunk = self.stt_ready_chunks.popleft() if self.stt_ready_chunks else None
        out_chunk = self.agent.process_audio(chunk, next_stt_chunk)
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
        return AgentHandler(self.agent, self.kyutai_stt_ws_url, self.kyutai_stt_api_key, self.report_interval_secs)

    def shutdown(self):
        if not self.started:
            print(">>> Not Started <<<")
            return
        transcript = self.agent.format_transcript()
        audio_first_sequence, text_first_sequence = self.agent.get_sequence_strs()
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- Transcript:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(transcript)
            f.write("\n\n")
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

        if self.stt_websocket is not None:
            self.stt_websocket.close()
            self.stt_websocket = None
        
        self.started = False
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        self.set_config_and_reset()
        if self.agent.config.audio_first_trans_mode == "none":
            self.stt_websocket = connect(self.kyutai_stt_ws_url, additional_headers={"kyutai-api-key": self.kyutai_stt_api_key})
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
            config.text_first_audio_temperature = float(self.latest_args[5])
            config.text_first_audio_top_p = float(self.latest_args[6])
            config.text_first_audio_min_p = float(self.latest_args[7])
            config.text_first_audio_presence_penalty = float(self.latest_args[8])
            config.text_first_audio_frequency_penalty = float(self.latest_args[9])
            config.text_first_text_temperature = float(self.latest_args[10])
            config.audio_first_trans_mode = self.latest_args[11]
            config.audio_first_cont_temperature = float(self.latest_args[12])
            config.audio_first_trans_temperature = float(self.latest_args[13])
            config.max_context_secs = float(self.latest_args[14])
            config.trim_by_secs = float(self.latest_args[15])
            config.target_volume_rms = float(self.latest_args[16])
            config.non_activity_limit_secs = float(self.latest_args[17])
            config.use_external_llm = bool(self.latest_args[18])
            config.external_llm_instructions = self.latest_args[19]

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
    )
    handler = AgentHandler(agent, args.kyutai_stt_ws_url, args.kyutai_stt_api_key)
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
            gr.Textbox(label="Agent Voice Enrollment Text"),
            gr.Slider(0.02, 1.0, value=0.1, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Text-First Audio Temperature"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Text-First Audio Top-p"),
            gr.Slider(0.0, 1.0, value=0.002, step=0.001, label="Text-First Audio Min-p"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Text-First Audio Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Text-First Audio Frequency Penalty"),
            gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Text-First Text Temperature"),
            gr.Dropdown(AUDIO_FIRST_TRANSCRIPTION_MODES, value=agent.config.audio_first_trans_mode, label="Audio-First Transcription Mode"),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Audio-First Continuation Temperature"),
            gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Audio-First Transcription Temperature"),
            gr.Number(40.0, minimum=5.0, maximum=80.0, step=5.0, label="Max Context Length (seconds)"),
            gr.Number(10.0, minimum=1.0, maximum=20.0, step=1.0, label="Trim By (seconds)"),
            gr.Slider(0.0, 0.1, value=0.05, step=0.01, label="Volume Normalization (0 to disable)"),
            gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Non-activity limit (seconds, 0 to disable)"),
            gr.Checkbox(False, label=f"Use External LLM ({agent.config.external_llm_model})"),
            gr.TextArea(label="External LLM Instructions"),
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
    parser.add_argument("--kyutai_stt_ws_url", help="Websocket URL for Kyutai STT Rust server", default="ws://127.0.0.1:8080/api/asr-streaming")
    parser.add_argument("--kyutai_stt_api_key", help="API Key for Kyutai STT Rust server", default="public_token")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv(override=True)

    main(args)