import argparse
import gradio as gr
import numpy as np
import itertools

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.audio_utils import smooth_join, create_crossfade_ramps
from codec_bpe import codes_to_chars, UNICODE_OFFSET_LARGE

audio_tokenizer = None
crossfade_ramps = None

def stream_codes(audio, codes_file, audio_str, audio_str_is_stereo, encoding_chunk_size_secs, decoding_chunk_size_secs, length_secs):
    audio_tokenizer.reset_context()

    if not audio and not codes_file and not audio_str:
        raise ValueError("Either audio or codes_file or audio_str must be provided.")
    if (audio and codes_file) or (audio and audio_str) or (codes_file and audio_str):
        raise ValueError("Only one of audio, codes_file or audio_str should be provided.")
    
    if audio_str:
        # get rid of any non-audio characters
        audio_str = "".join([ch for ch in audio_str if ord(ch) >= UNICODE_OFFSET_LARGE])

    mono_input = False
    if audio:
        # stream-encode the audio to codes
        sr, audio = audio
        if audio.ndim == 2:
            audio = audio.T
        else:
            audio = np.stack([audio, audio])
            mono_input = True
        chunk_size_samples = int(encoding_chunk_size_secs * sr)
        encode_samples = min(int(length_secs * sr), audio.shape[-1]) if length_secs > 0 else audio.shape[-1]
        audio_str = ""
        for start in range(0, encode_samples, chunk_size_samples):
            end = start + chunk_size_samples
            chunk = audio[..., start:end]
            chunk_str = audio_tokenizer.tokenize_audio((sr, chunk))
            audio_str += chunk_str
    elif codes_file:
        codes = np.load(codes_file)
        if codes.shape[0] == 1:
            codes = np.concatenate([codes, codes], axis=0)
            mono_input = True
        channels_chars = [
            codes_to_chars(
                ch_codes, 
                audio_tokenizer.codebook_size, 
                unicode_offset=UNICODE_OFFSET_LARGE,
            ) for ch_codes in codes
        ]
        audio_str = "".join(list(itertools.chain.from_iterable(zip(*channels_chars))))
    elif audio_str and not audio_str_is_stereo:
        channels_chars = [audio_str, audio_str]
        audio_str = "".join(list(itertools.chain.from_iterable(zip(*channels_chars))))
        mono_input = True

    # stream-decode the reconstruction
    chunk_size_frames = int(decoding_chunk_size_secs * audio_tokenizer.framerate * audio_tokenizer.num_channels)
    decode_frames = min(int(length_secs * audio_tokenizer.framerate * audio_tokenizer.num_channels), len(audio_str)) if length_secs > 0 else len(audio_str)
    audio = np.zeros((audio_tokenizer.num_channels, 0), dtype=np.float32)
    for start in range(0, decode_frames, chunk_size_frames):
        end = start + chunk_size_frames
        chunk = audio_str[start:end]
        (_, output_audio), _, _ = audio_tokenizer.detokenize_audio(chunk, preroll_samples=crossfade_ramps[0])
        audio = smooth_join(audio, output_audio.reshape(audio_tokenizer.num_channels, -1), *crossfade_ramps)
    
    audio = audio[0] if mono_input else audio.T
    return audio_tokenizer.sampling_rate, audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming codes interface")
    parser.add_argument("--codec_model", type=str, default="MagiCodec-50Hz-Base")
    parser.add_argument("--context_secs", type=float, default=2.0, help="Context size in seconds for the audio tokenizer")
    
    args = parser.parse_args()

    print(f"Running with args: {args}")

    audio_tokenizer = AudioTokenizer(codec_model=args.codec_model, num_channels=2, context_secs=args.context_secs)
    crossfade_ramps = create_crossfade_ramps(audio_tokenizer.sampling_rate, fade_secs=0.02)

    interface = gr.Interface(
        fn=stream_codes,
        inputs=[
            gr.Audio(label="Audio"),
            gr.Textbox(label="Codes file"),
            gr.Textbox(label="Codes string"),
            gr.Checkbox(label="Codes string is stereo", value=False),
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