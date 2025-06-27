import gradio as gr
import argparse
import logging
import re
import numpy as np
from openai import OpenAI

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.vllm_utils import get_vllm_modelname
from run_demo import prep_for_output

logger = logging.getLogger(__name__)

client = None
model_name = None

audio_tokenizer = None

def generate_audio(
    context_audio,
    transcript, 
    show_all_codes,
    seed, 
    temperature, 
    top_p, 
    min_p,
):
    audio_tokenizer.reset_context()
    
    if context_audio is not None:
        if context_audio[1].ndim == 2:
            context_audio = (context_audio[0], context_audio[1].T)
        input_audio_str = audio_tokenizer.tokenize_audio(context_audio)

    num_speakers = len(set(re.findall("[A-Z]: ", transcript)))
    transcript_lines = [l.strip() for l in transcript.split("\n")]
    
    sequence = "<|text_first|>"
    sequence += "".join([f"<|speaker|>{chr(ord('A') + i % 26)}" for i in range(num_speakers)])
    sequence += "<|end_header|>"
    if context_audio is not None:
        sequence += "<|audio|>" + input_audio_str + "<|end_audio|>"

    extra_body = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if float(min_p) > 0.0:
        extra_body["min_p"] = float(min_p)
    transcription_audio = np.zeros((audio_tokenizer.num_channels, 0), dtype=np.float32)
    end_hanging = ""
    for line in transcript_lines:
        sequence += f" {line}<|audio|>{end_hanging}"

        completion = client.completions.create(
            model=model_name, 
            prompt=sequence, 
            seed=int(seed) if seed else None,
            max_tokens=1024*audio_tokenizer.num_channels, 
            temperature=float(temperature),
            top_p=float(top_p),
            extra_body=extra_body,
            stream=False,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            stop="<|end_audio|>",
        )
        (_, audio_chunk), end_hanging, _ = audio_tokenizer.detokenize_audio(end_hanging + completion.choices[0].text)
        sequence += completion.choices[0].text[:(-len(end_hanging) or None)] + "<|end_audio|>"
        transcription_audio = np.concatenate((transcription_audio, audio_chunk.reshape(audio_tokenizer.num_channels, -1)), axis=-1)
        outputs = prep_for_output(audio_chunk, transcription_audio, audio_tokenizer.sampling_rate, sequence, show_all_codes)
        yield outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming TTS demo")
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
            gr.Audio(label="Audio Context"),
            gr.Textbox(label="Transcript", max_lines=20),
            gr.Checkbox(False, label="Show all codes"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Slider(0.1, 10.0, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
        ], 
        outputs=[
            gr.Audio(label="Streaming Audio", streaming=True, autoplay=True),
            gr.Audio(label="Audio"),
            gr.TextArea(label="Streaming Transcript")
        ],
        allow_flagging='never'
    )
    interface.launch()