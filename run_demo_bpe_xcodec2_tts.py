import torch
import gradio as gr
import argparse
import librosa
import logging
import re
import numpy as np
from openai import OpenAI
from codec_bpe import codes_to_chars, chars_to_codes, UNICODE_OFFSET_LARGE
from xcodec2.modeling_xcodec2 import XCodec2Model

from realtime_codec_agent.utils.vllm_utils import get_vllm_modelname

logger = logging.getLogger(__name__)

device = None
client = None
model_name = None

codec_model = None

shorten_codes_regex = r"(?<=[^>]{4})[^<>]+(?=[^<]{4}<\|end_audio\|>)"

def tokenize_audio(audio):
    orig_sr, audio = audio
    audio = audio.astype("float32") / 32768.0
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    sr = codec_model.feature_extractor.sampling_rate
    # resample to codec sample rate if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    audio = torch.tensor(audio).unsqueeze(0).to(device)

    # get audio codes
    encoder_outputs = codec_model.encode_code(audio, sample_rate=sr)

    # convert to unicode string
    audio_codes_str = codes_to_chars(
        encoder_outputs[0],
        65536,
        unicode_offset=UNICODE_OFFSET_LARGE,
    )

    return audio_codes_str

def detokenize_audio(audio_codes_str):
    # convert unicode string to audio codes
    audio_codes, _, end_hanging = chars_to_codes(
        audio_codes_str, 
        1, 
        65536,
        return_hanging_codes_chars=True, 
        return_tensors="pt",
        unicode_offset=UNICODE_OFFSET_LARGE,
    )
    audio_codes = audio_codes.unsqueeze(0).to(device)
    
    # decode audio codes with codec
    with torch.no_grad():
        output_audio = codec_model.decode_code(audio_codes)

    # return audio
    sr = codec_model.feature_extractor.sampling_rate
    return (sr, output_audio[0, 0].cpu().numpy()), end_hanging

def generate_audio(
    context_audio,
    transcript, 
    show_all_codes,
    seed, 
    decoding, 
    temperature, 
    num_beams, 
    top_k, 
    top_p, 
    min_p,
    typical_p, 
):
    generate_kwargs = {
        "num_beams": int(num_beams),
        "early_stopping": True,
        "do_sample": False
    }
    if decoding != "Greedy":
        generate_kwargs["top_k"] = int(top_k)
    if decoding == "Sampling":
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
        generate_kwargs["typical_p"] = float(typical_p)
        if float(min_p) > 0.0:
            generate_kwargs["min_p"] = float(min_p)

    extra_body = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if "min_p" in generate_kwargs:
        extra_body["min_p"] = generate_kwargs["min_p"]
    
    if context_audio is not None:
        input_audio_str = tokenize_audio(context_audio)

    num_speakers = len(set(re.findall("[A-Z]: ", transcript)))
    transcript_lines = [l.strip() for l in transcript.split("\n")]
    
    sequence = "<|text_first|>"
    sequence += "".join([f"<|speaker|>{chr(ord('A') + i % 26)}" for i in range(num_speakers)])
    sequence += "<|end_header|>"
    if context_audio is not None:
        sequence += "<|audio|>" + input_audio_str + "<|end_audio|>"

    codec_framerate = 50.0
    audio_chunks = []
    audio_str = ""
    end_hanging = ""
    for line in transcript_lines:
        sequence += f" {line}<|audio|>"

        completion = client.completions.create(
            model=model_name, 
            prompt=sequence, 
            seed=int(seed) if seed else None,
            max_tokens=1024, 
            temperature=generate_kwargs["temperature"],
            top_p=generate_kwargs["top_p"],
            best_of=generate_kwargs["num_beams"],
            extra_body=extra_body,
            stream=False,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            stop="<|end_audio|>",
        )
        audio_chars = end_hanging + completion.choices[0].text
        chunk_duration = len(audio_chars) / codec_framerate
        audio_str += audio_chars
        audio_chunk, end_hanging = detokenize_audio(audio_str)
        audio_chunk = (audio_chunk[0], audio_chunk[1][..., -int(chunk_duration*audio_chunk[0]):])
        audio_str = audio_str[-int(3*codec_framerate):]
        audio_chunks.append(audio_chunk)
        sequence += audio_chars + "<|end_audio|>"
        if not show_all_codes:
            output_transcript = re.sub(shorten_codes_regex, ".........", sequence)
        else:
            output_transcript = sequence

        full_audio = (audio_chunks[0][0], np.concatenate([audio[1] for audio in audio_chunks], axis=-1))
        yield audio_chunk, full_audio, output_transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming TTS demo")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1")
    args = parser.parse_args()

    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vllm_api_key = "Empty"
    client = OpenAI(
        api_key=vllm_api_key,
        base_url=args.vllm_base_url,
    )
    model_name = get_vllm_modelname(args.vllm_base_url, vllm_api_key)
    if model_name is None:
        raise ValueError("Could not find a model hosted by the vLLM server.")
 
    codec_model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2")
    codec_model.eval().to(device)

    interface = gr.Interface(
        fn=generate_audio,
        inputs=[
            gr.Audio(label="Audio Context"),
            gr.Textbox(label="Transcript", max_lines=20),
            gr.Checkbox(False, label="Show all codes"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(0.1, 10.0, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(1, 100, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=0, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
        ], 
        outputs=[
            gr.Audio(label="Streaming Audio", streaming=True, autoplay=True),
            gr.Audio(label="Audio"),
            gr.TextArea(label="Streaming Transcript")
        ],
        allow_flagging='never'
    )
    interface.launch()