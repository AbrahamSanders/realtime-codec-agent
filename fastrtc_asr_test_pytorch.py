from fastrtc import StreamHandler, Stream, AdditionalOutputs
from queue import Queue, Empty
from datetime import datetime
import numpy as np
import gradio as gr
import torch


from dataclasses import dataclass
import sentencepiece
from moshi.models import loaders, MimiModel, LMModel, LMGen

@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(
        self,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        device: str | torch.device,
    ):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        self.first_frame = True

        # Warm up the mimi model, this avoids some CUDA Graph related errors
        # when running the model for the first time.
        with torch.inference_mode():
            for _ in range(2):
                mimi.encode(torch.zeros((1, 1, mimi.frame_size), device=device))

    @torch.inference_mode()
    def run(self, chunk: torch.Tensor):
        codes = self.mimi.encode(chunk)
        if self.first_frame:
            # Ensure that the first slice of codes is properly seen by the transformer
            # as otherwise the first slice is replaced by the initial tokens.
            tokens = self.lm_gen.step(codes)
            self.first_frame = False
        tokens = self.lm_gen.step(codes)
        if tokens is None:
            return None
        assert tokens.shape[1] == 1
        one_text = tokens[0, 0].cpu()
        if one_text.item() not in [0, 3]:
            text = self.text_tokenizer.id_to_piece(one_text.item())
            text = text.replace("▁", " ")
            return text
        return None

class ASRStreamHandler(StreamHandler):
    def __init__(self, state: InferenceState) -> None:
        super().__init__(
            input_sample_rate=24000,
            output_sample_rate=24000,
        )
        self.state = state
        self.state.first_frame = True
        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.queue = Queue()
        self.transcript = ""

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, frame_audio = frame
        chunk_size_samples = int(self.input_sample_rate * 0.08)
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [chunk_size_samples], axis=1)
            self.queue.put(chunk)

    def emit(self) -> None:
        try:
            chunk = self.queue.get_nowait()
        except Empty:
            return None
        
        start_time = datetime.now()
        chunk = chunk.astype(np.float32) / 32768.0
        chunk = torch.from_numpy(chunk).unsqueeze(0).to(self.state.device)
        text = self.state.run(chunk)
        end_time = datetime.now()
        elapsed_secs = (end_time - start_time).total_seconds()
        realtime_factor = 0.08 / elapsed_secs
        
        if text is not None:
            self.transcript += text
        return AdditionalOutputs(f"{realtime_factor:.2f}x", text, self.transcript)

    def copy(self):
        return ASRStreamHandler(self.state)

    def shutdown(self):
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        print(">>> Started <<<")

def transcription_handler(component1, component2, component3, realtime_factor: str, trans_chunk: str, transcript: str):
    return realtime_factor, trans_chunk, transcript

def main():
    device = "cuda"
    # Use the en+fr low latency model, an alternative is kyutai/stt-2.6b-en
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    lm = checkpoint_info.get_moshi(device=device)
    state = InferenceState(mimi, text_tokenizer, lm, batch_size=1, device=device)

    handler = ASRStreamHandler(state)
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
            gr.TextArea(),
        ],
        additional_outputs_handler=transcription_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    main()