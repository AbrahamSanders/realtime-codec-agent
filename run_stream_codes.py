import argparse
import gradio as gr
import numpy as np

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.audio_utils import smooth_join, create_crossfade_ramps
from codec_bpe import codes_to_chars, chars_to_codes, UNICODE_OFFSET_LARGE

audio_tokenizer = None
crossfade_ramps = None

def stream_codes(audio, codes_file, encoding_chunk_size_secs, decoding_chunk_size_secs, length_secs):
    audio_tokenizer.reset_context()

    if not audio and not codes_file:
        raise ValueError("Either audio or codes_file must be provided.")
    if audio and codes_file:
        raise ValueError("Only one of audio or codes_file should be provided.")
    if audio:
        # stream-encode the audio to codes
        sr, audio = audio
        if audio.ndim == 2:
            audio = audio.T
        chunk_size_samples = int(encoding_chunk_size_secs * sr)
        encode_samples = min(int(length_secs * sr), audio.shape[-1]) if length_secs > 0 else audio.shape[-1]
        audio_str = ""
        for start in range(0, encode_samples, chunk_size_samples):
            end = start + chunk_size_samples
            chunk = audio[..., start:end]
            chunk_str = audio_tokenizer.tokenize_audio((sr, chunk))
            audio_str += chunk_str
        codes = chars_to_codes(
            audio_str, 
            audio_tokenizer.num_codebooks, 
            audio_tokenizer.codebook_size, 
            return_tensors="np", 
            unicode_offset=UNICODE_OFFSET_LARGE,
        )
        codes = np.expand_dims(codes, axis=0)
    elif codes_file:
        codes = np.load(codes_file)

    # stream-decode the reconstruction
    chunk_size_frames = int(decoding_chunk_size_secs * audio_tokenizer.framerate)
    decode_frames = min(int(length_secs * audio_tokenizer.framerate), codes.shape[-1]) if length_secs > 0 else codes.shape[-1]
    audio = np.zeros((audio_tokenizer.num_channels, 0), dtype=np.float32)
    for start in range(0, decode_frames, chunk_size_frames):
        end = start + chunk_size_frames
        chunk = codes[..., start:end]
        audio_str = codes_to_chars(chunk[0], audio_tokenizer.codebook_size, unicode_offset=UNICODE_OFFSET_LARGE)
        (_, output_audio), _, _ = audio_tokenizer.detokenize_audio(audio_str, preroll_samples=crossfade_ramps[0])
        audio = smooth_join(audio, output_audio.reshape(audio_tokenizer.num_channels, -1), *crossfade_ramps)
    return audio_tokenizer.sampling_rate, audio[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming codes interface")
    parser.add_argument("--codec_model", type=str, default="MagiCodec-50Hz-Base")
    parser.add_argument("--context_secs", type=float, default=2.0, help="Context size in seconds for the audio tokenizer")
    
    args = parser.parse_args()

    print(f"Running with args: {args}")

    audio_tokenizer = AudioTokenizer(codec_model=args.codec_model, context_secs=args.context_secs)
    crossfade_ramps = create_crossfade_ramps(audio_tokenizer.sampling_rate, fade_secs=0.02)

    interface = gr.Interface(
        fn=stream_codes,
        inputs=[
            gr.Audio(label="Audio"),
            gr.Textbox(label="Codes file"),
            gr.Slider(label="Encoding chunk size (seconds)", minimum=0.02, maximum=1.0, value=0.1, step=0.02),
            gr.Slider(label="Decoding chunk size (seconds)", minimum=0.02, maximum=1.0, value=0.1, step=0.02),
            gr.Slider(label="Length (seconds): zero for full length", minimum=0, maximum=120, value=30, step=1),
        ], 
        outputs=[
            gr.Audio(label="Audio"),
        ],
        allow_flagging='never',
    )
    interface.launch()