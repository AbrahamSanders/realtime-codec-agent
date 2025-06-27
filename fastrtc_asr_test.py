from fastrtc import StreamHandler, Stream, AdditionalOutputs
import numpy as np
import gradio as gr
import re

from realtime_codec_agent.asr_handler import ASRHandlerMultiprocessing, ASRConfig

class ASRStreamHandler(StreamHandler):
    def __init__(self, asr_handler) -> None:
        super().__init__(
            input_sample_rate=16000,
            output_sample_rate=16000,
        )
        self.asr_handler = asr_handler
        self.in_buffer = np.zeros((1, 0), dtype=np.int16)
        self.transcript = ""
        self.partial_pos = -1

    def update_transcription(self, transcription, new_text, partial_pos):
        if new_text:
            # First, clear out the previous partial segment (if exists)
            if partial_pos > -1:
                transcription = transcription[:partial_pos]
                partial_pos = -1
            # Next, add the new segments to the transcription, 
            # discarding intermediate partial segments.
            new_segments = re.split(" (?=[~*])", new_text)
            for i, seg in enumerate(new_segments):
                if len(seg) > 1 and (seg.startswith("*") or i == len(new_segments)-1):
                    if seg.startswith("~"):
                        partial_pos = len(transcription)
                    if len(transcription) > 0:
                        transcription += " "
                    transcription += seg[1:]
        return transcription, partial_pos

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, frame_audio = frame
        chunk_size_samples = int(self.input_sample_rate * 0.4)
        self.in_buffer = np.concatenate((self.in_buffer, frame_audio), axis=1)
        if self.in_buffer.shape[-1] >= chunk_size_samples:
            chunk, self.in_buffer = np.split(self.in_buffer, [chunk_size_samples], axis=1)
            self.asr_handler.queue_input((self.input_sample_rate, chunk.squeeze(0)))

    def emit(self) -> None:
        trans_chunk = self.asr_handler.next_output()
        if not trans_chunk:
            return None
        
        self.transcript, self.partial_pos = self.update_transcription(self.transcript, trans_chunk, self.partial_pos)
        return AdditionalOutputs(trans_chunk, self.transcript)

    def copy(self):
        return ASRStreamHandler(self.asr_handler)

    def shutdown(self):
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        print(">>> Started <<<")

def transcription_handler(component1, component2, trans_chunk: str, transcript: str):
    return trans_chunk, transcript

def main():
    asr_handler = ASRHandlerMultiprocessing(
        config = ASRConfig(
            model_size="small.en",
            n_context_segs=2,
            n_prefix_segs=2,
            max_buffer_size=10,
        ),
    )

    handler = ASRStreamHandler(asr_handler)
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
            gr.Textbox(),
            gr.TextArea()
        ],
        additional_outputs_handler=transcription_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    main()