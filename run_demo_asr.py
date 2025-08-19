import gradio as gr
import argparse
import logging
from openai import OpenAI

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.vllm_utils import get_vllm_modelname
from run_demo import prep_for_output

logger = logging.getLogger(__name__)

client = None
model_name = None

audio_tokenizer = None

def generate_transcript(
    audio, 
    num_speakers,
    show_all_codes,
    seed, 
    cont_temperature, 
    trans_temperature,
    top_p, 
    min_p, 
):
    audio_tokenizer.reset_context()
    sr, audio = audio
    if audio.ndim > 1:
        audio = audio.T
    
    sequence = "<|audio_first|>"
    sequence += "".join([f"<|speaker|> {chr(ord('A') + i % 26)}" for i in range(num_speakers)])
    sequence += "<|end_header|>"
    sequence += "<|audio|>"

    extra_body = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    if float(min_p) > 0.0:
        extra_body["min_p"] = float(min_p)
    chunk_size_secs = 0.1
    chunk_size_samples = int(chunk_size_secs*sr)
    transcribe_after = None
    start = 0
    while start < audio.shape[-1]:
        # tokenize next audio chunk
        audio_chunk = (sr, audio[..., start:start+chunk_size_samples])
        input_audio_str = audio_tokenizer.tokenize_audio(audio_chunk)

        if transcribe_after is not None:
            sequence += f"{input_audio_str[:transcribe_after]}<|end_audio|>"

            # transcribe the audio
            completion = client.completions.create(
                model=model_name, 
                prompt=sequence, 
                seed=int(seed) if seed else None,
                max_tokens=100, 
                temperature=float(trans_temperature),
                top_p=float(top_p),
                extra_body=extra_body,
                stream=False,
                stop="<|audio|>",
            )
            completion_text = completion.choices[0].text
            sequence += f"{completion_text}<|audio|>{input_audio_str[transcribe_after:]}"
            transcribe_after = None
        else:
            sequence += input_audio_str

            # decide if it's time to transcribe by generating the next chunk and checking if the model predicts
            # a transcription within it
            completion = client.completions.create(
                model=model_name, 
                prompt=sequence, 
                seed=int(seed) if seed else None,
                max_tokens=int(chunk_size_secs * audio_tokenizer.framerate * audio_tokenizer.num_channels), 
                temperature=float(cont_temperature),
                top_p=float(top_p),
                extra_body=extra_body,
                stream=False,
                stop="<|end_audio|>",
            )
            if completion.choices[0].finish_reason == "stop":
                completion_text = completion.choices[0].text
                transcribe_after = len(completion_text)
                div_rem = transcribe_after % audio_tokenizer.num_channels
                if div_rem != 0:
                    transcribe_after -= div_rem

        audio_chunk, _, sequence_display = prep_for_output(audio_chunk[1], None, sr, sequence, show_all_codes)
        yield audio_chunk, sequence_display
        start += chunk_size_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming ASR demo")
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
        fn=generate_transcript,
        inputs=[
            gr.Audio(label="Audio"),
            gr.Slider(1, 10, value=2, step=1, label="Number of speakers"),
            gr.Checkbox(False, label="Show all codes"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Slider(0.0, 10.0, value=0.6, step=0.01, label="Continuation Temperature"),
            gr.Slider(0.0, 10.0, value=0.2, step=0.01, label="Transcription Temperature"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
        ], 
        outputs=[
            gr.Audio(label="Streaming Audio", streaming=True, autoplay=True),
            gr.TextArea(label="Streaming Transcript")
        ],
        allow_flagging='never'
    )
    interface.launch()