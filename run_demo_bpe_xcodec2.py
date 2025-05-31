import gradio as gr
import argparse
import logging
import re
import numpy as np
from openai import OpenAI

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.vllm_utils import get_vllm_modelname

logger = logging.getLogger(__name__)

client = None
model_name = None

audio_tokenizer = None

shorten_codes_regex = r"(?<=[^>]{4})[^<>]+(?=[^<]{4}<\|end_audio\|>)"

def prep_for_output(output_audio, completion_audio, sr, completion_text, show_all_codes):
    if output_audio.ndim == 2:
        output_audio = output_audio.T
        if completion_audio is not None:
            completion_audio = completion_audio.T
    elif completion_audio is not None and completion_audio.ndim == 2:
        completion_audio = completion_audio[0]

    if not show_all_codes:
        completion_text = re.sub(shorten_codes_regex, ".........", completion_text)

    return (sr, output_audio), (sr, completion_audio), completion_text

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
    seconds, 
    temperature, 
    top_p, 
    min_p,
):
    audio_tokenizer.reset_context()

    if not text_prompt_only:
        if context_audio[1].ndim == 2:
            context_audio = (context_audio[0], context_audio[1].T)
        input_audio_str = audio_tokenizer.tokenize_audio(context_audio)
        input_audio_str = f"<|audio|>{input_audio_str}<|end_audio|>"

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
    if float(min_p) > 0.0:
        extra_body["min_p"] = float(min_p)
        
    completion = client.completions.create(
        model=model_name, 
        prompt=model_inputs, 
        seed=int(seed) if seed else None,
        max_tokens=int(seconds * audio_tokenizer.framerate * audio_tokenizer.num_channels), 
        temperature=float(temperature),
        top_p=float(top_p),
        extra_body=extra_body,
        stream=True,
    )

    completion_text = ""
    completion_audio = np.zeros((audio_tokenizer.num_channels, 0), dtype=np.float32)
    audio_str = ""
    for chunk in completion:
        chunk_text = chunk.choices[0].text
        audio_str += "".join([c for c in chunk_text if ord(c) >= audio_tokenizer.unicode_offset])
        completion_text += chunk_text
        audio_str_secs = audio_tokenizer.get_audio_codes_str_secs(audio_str)
        if audio_str_secs >= 2.0:
            (_, output_audio), audio_str, _ = audio_tokenizer.detokenize_audio(audio_str)
            completion_audio = np.concatenate((completion_audio, output_audio.reshape(audio_tokenizer.num_channels, -1)), axis=-1)
            outputs = prep_for_output(output_audio, completion_audio, audio_tokenizer.sampling_rate, completion_text, show_all_codes)
            yield outputs
    if len(audio_str) > 0:
        (_, output_audio), _, _ = audio_tokenizer.detokenize_audio(audio_str)
        completion_audio = np.concatenate((completion_audio, output_audio.reshape(audio_tokenizer.num_channels, -1)), axis=-1)
        outputs = prep_for_output(output_audio, completion_audio, audio_tokenizer.sampling_rate, completion_text, show_all_codes)
        yield outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the audio generator demo")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--stereo", action="store_true", help="Use interleaved two-channel code sequences")
    args = parser.parse_args()

    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    vllm_api_key = "Empty"
    client = OpenAI(
        api_key=vllm_api_key,
        base_url=args.vllm_base_url,
    )
    model_name = get_vllm_modelname(args.vllm_base_url, vllm_api_key)
    if model_name is None:
        raise ValueError("Could not find a model hosted by the vLLM server.")
    
    if "stereo" in model_name.lower():
        args.stereo = True
        print("Model is a stereo model, setting --stereo to true")
 
    audio_tokenizer = AudioTokenizer(num_channels = 2 if args.stereo else 1)

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
            gr.Slider(1, 60, value=30, step=1, label="Seconds"),
            gr.Slider(0.1, 10.0, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
        ], 
        outputs=[
            gr.Audio(label="Continuation (streaming)", streaming=True, autoplay=True),
            gr.Audio(label="Continuation"),
            gr.TextArea(label="Continuation (Text)")
        ],
        allow_flagging='never'
    )
    interface.launch()