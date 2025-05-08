import torch
import gradio as gr
import argparse
import librosa
import logging
import numpy as np
import re
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
        audio = librosa.to_mono(audio.T)
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

    return f"<|audio|>{audio_codes_str}<|end_audio|>"

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
        text_prompt, 
        num_speakers,
        audio_prompt_only,
        text_prompt_only,
        text_prompt_first,
        interleave_text_first,
        show_all_codes,
        seed, 
        decoding, 
        seconds, 
        temperature, 
        num_beams, 
        top_k, 
        top_p, 
        min_p,
        typical_p, 
    ):
    codec_framerate = 50.0
    generate_kwargs = {
        "num_beams": int(num_beams),
        "early_stopping": True,
        "max_new_tokens": int(seconds * codec_framerate),
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

    if not text_prompt_only:
        input_audio_str = tokenize_audio(context_audio)

    if audio_prompt_only and text_prompt_only:
        raise ValueError("audio_prompt_only and text_prompt_only cannot both be True.")
    elif not audio_prompt_only and not text_prompt_only:
        model_inputs = " " + text_prompt + input_audio_str if text_prompt_first else input_audio_str + " " + text_prompt + "<|audio|>"
    elif audio_prompt_only:
        model_inputs = input_audio_str
    else:
        model_inputs = " " + text_prompt + "<|audio|>"
    
    header = "<|text_first|>" if interleave_text_first else "<|audio_first|>"
    header += "".join([f"<|speaker|>{chr(ord('A') + i % 26)}" for i in range(num_speakers)])
    header += "<|end_header|>"

    model_inputs = header + model_inputs
    print(model_inputs)

    extra_body = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if "min_p" in generate_kwargs:
        extra_body["min_p"] = generate_kwargs["min_p"]
    stream = generate_kwargs["num_beams"]==1
    completion = client.completions.create(
        model=model_name, 
        prompt=model_inputs, 
        seed=int(seed) if seed else None,
        max_tokens=generate_kwargs["max_new_tokens"], 
        temperature=generate_kwargs["temperature"],
        top_p=generate_kwargs["top_p"],
        best_of=generate_kwargs["num_beams"],
        extra_body=extra_body,
        stream=stream,
        # presence_penalty=0.25,
        # frequency_penalty=0.25,
    )
    if not stream:
        completion = [completion]
    chunk_duration = 0.
    chunk_audio_strs = []
    text_chunks = []
    audio_chunks = []
    end_hanging = ""
    audio_str = ""
    for chunk in completion:
        chunk_text = chunk.choices[0].text
        chunk_audio_str = "".join([c for c in chunk_text if ord(c) >= UNICODE_OFFSET_LARGE])
        if chunk_text:
            text_chunks.append(chunk_text)
        if chunk_audio_str:
            chunk_units = len(chunk_audio_str)
            chunk_secs = chunk_units / codec_framerate
            chunk_duration += chunk_secs
            chunk_audio_strs.append(chunk_audio_str)
            if chunk_duration >= 2.0:
                audio_str += end_hanging + "".join(chunk_audio_strs)
                output_audio, end_hanging = detokenize_audio(audio_str)
                output_audio = (output_audio[0], output_audio[1][..., -int(chunk_duration*output_audio[0]):])
                audio_str = audio_str[-int(3*codec_framerate):]
                chunk_audio_strs.clear()
                chunk_duration = 0.
                audio_chunks.append(output_audio)
                text_chunks_concat = "".join(text_chunks)
                if not show_all_codes:
                    text_chunks_concat = re.sub(shorten_codes_regex, ".........", text_chunks_concat)
                yield output_audio, (audio_chunks[0][0], np.concatenate([audio[1] for audio in audio_chunks], axis=-1)), text_chunks_concat
    if len(chunk_audio_strs) > 0:
        audio_str += end_hanging + "".join(chunk_audio_strs)
        output_audio, _ = detokenize_audio(audio_str)
        output_audio = (output_audio[0], output_audio[1][..., -int(chunk_duration*output_audio[0]):])
        audio_chunks.append(output_audio)
        text_chunks_concat = "".join(text_chunks)
        if not show_all_codes:
            text_chunks_concat = re.sub(shorten_codes_regex, ".........", text_chunks_concat)
        yield output_audio, (audio_chunks[0][0], np.concatenate([audio[1] for audio in audio_chunks], axis=-1)), text_chunks_concat


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the audio generator demo")
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
            gr.Audio(label="Context"),
            gr.Textbox(label="Text prompt"),
            gr.Slider(1, 10, value=2, step=1, label="Number of speakers"),
            gr.Checkbox(False, label="Audio prompt only"),
            gr.Checkbox(False, label="Text prompt only"),
            gr.Checkbox(False, label="Text prompt first"),
            gr.Checkbox(True, label="Interleave text first"),
            gr.Checkbox(False, label="Show all codes"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(1, 60, value=30, step=1, label="Seconds"),
            gr.Slider(0.1, 10.0, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(1, 100, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=0, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
        ], 
        outputs=[
            gr.Audio(label="Continuation (streaming)", streaming=True, autoplay=True),
            gr.Audio(label="Continuation"),
            gr.TextArea(label="Continuation (Text)")
        ],
        allow_flagging='never'
    )
    interface.launch()