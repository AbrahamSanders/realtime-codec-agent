import gradio as gr
from typing import Tuple
import numpy as np
from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.audio_utils import smooth_join, create_crossfade_ramps
from realtime_codec_agent.external_tts_client import ExternalTTSClient

audio_tokenizer: AudioTokenizer = None

def tts_pipeline(enrollment_audio_numpy, enrollment_prompt_text: str, target_text: str) -> Tuple[int, np.ndarray]:
    tts_client = ExternalTTSClient()
    tts_client.set_voice_enrollment(enrollment_audio_numpy, enrollment_prompt_text)
    target_texts = [t.strip() for t in target_text.split("\n") if t.strip()]
    audio_chunks = []
    crossfade_ramps = create_crossfade_ramps(audio_tokenizer.sampling_rate, fade_secs=0.02)
    for target_text in target_texts:
        tts_client.prep_stream(target_text)
        while True:
            chunk = tts_client.next_chunk()
            if chunk is None:
                break
            (_, chunk), _, _ = audio_tokenizer.detokenize_audio(chunk, preroll_samples=crossfade_ramps[0])
            if len(audio_chunks) > 0:
                chunk_len = audio_chunks[-1].shape[-1]
                joined_chunk = smooth_join(audio_chunks[-1], chunk, *crossfade_ramps)
                audio_chunks[-1] = joined_chunk[:chunk_len]
                chunk = joined_chunk[chunk_len:]
            audio_chunks.append(chunk)
    audio_chunks = np.concatenate(audio_chunks, axis=-1)
    return audio_tokenizer.sampling_rate, audio_chunks

demo = gr.Interface(
    fn=tts_pipeline,
    inputs=[
        gr.Audio(label="Enrollment Audio"),
        gr.Textbox(label="Enrollment Prompt Text (transcript of enrollment audio)", lines=2),
        gr.Textbox(label="Target Text to Synthesize", lines=4),
    ],
    outputs=[
        gr.Audio(label="Generated Audio"),
    ],
    title="TTS Client",
    description="Provide an enrollment audio + its transcript, then enter target text to get streamed codec chunk strings.",
)

if __name__ == "__main__":
    audio_tokenizer = AudioTokenizer()
    demo.launch(server_name="0.0.0.0", server_port=7860)