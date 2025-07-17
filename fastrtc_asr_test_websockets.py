from fastrtc import StreamHandler, Stream, AdditionalOutputs
from datetime import datetime
from websockets.sync.client import connect
import numpy as np
import gradio as gr
import msgpack

class ASRStreamHandler(StreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=24000,
            output_sample_rate=24000,
        )
        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.websocket = None
        self.transcript = ""
        self.timed_transcript = ""
        self.trans_chunk = ""
        self.trans_chunk_start = None
        self.timing = {}
        self.send_count = 0
        self.recv_count = 0

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if self.websocket is None:
            return
        _, frame_audio = frame
        chunk_size_samples = int(self.input_sample_rate * 0.08)
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [chunk_size_samples], axis=1)
            self.timing[self.send_count] = datetime.now()
            chunk = chunk.squeeze(0).astype(np.float32) / 32768.0
            chunk = {"type": "Audio", "pcm": chunk.tolist()}
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            self.websocket.send(msg)
            self.send_count += 1

    def emit(self) -> None:
        if self.websocket is None:
            return None
        try:
            message = self.websocket.recv(timeout=0)
        except TimeoutError:
            return None
        
        data = msgpack.unpackb(message, raw=False)
        
        text = ""
        pause = data["type"] == "Step" and data["prs"][2] > 0.99 and self.trans_chunk_start is not None
        if data["type"] == "Word":
            text = data["text"]
            self.transcript += f" {text}"
            self.trans_chunk += f" {text}"
            if self.trans_chunk_start is None:
                self.trans_chunk_start = data["start_time"]

        if pause or text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith(","):
            self.timed_transcript += f"{self.trans_chunk_start}:{self.trans_chunk}\n"
            self.trans_chunk = ""
            self.trans_chunk_start = None

        realtime_factor = -1
        if self.recv_count in self.timing:
            start_time = self.timing[self.recv_count]
            end_time = datetime.now()
            elapsed_secs = (end_time - start_time).total_seconds()
            realtime_factor = 0.08 / elapsed_secs
            del self.timing[self.recv_count]
            self.recv_count += 1

        return AdditionalOutputs(f"{realtime_factor:.2f}x", text, self.transcript, self.timed_transcript)

    def copy(self):
        return ASRStreamHandler()

    def shutdown(self):
        if self.websocket is not None:
            self.websocket.close()
            self.websocket = None
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        self.websocket = connect("ws://127.0.0.1:8080/api/asr-streaming", additional_headers={"kyutai-api-key": "public_token"})
        print(">>> Started <<<")

def transcription_handler(c1, c2, c3, c4, realtime_factor: str, text: str, transcript: str, timed_transcript: str):
    return realtime_factor, text, transcript, timed_transcript

def main():
    handler = ASRStreamHandler()
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
        additional_outputs=[
            gr.Textbox(label="Realtime Factor"),
            gr.Textbox(),
            gr.TextArea(label="Transcript"),
            gr.TextArea(label="Timed Transcript"),
        ],
        additional_outputs_handler=transcription_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    main()