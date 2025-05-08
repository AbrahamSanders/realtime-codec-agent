import torch
import gradio as gr
import argparse
import librosa
import logging
import re
from openai import OpenAI
from codec_bpe import codes_to_chars, UNICODE_OFFSET_LARGE
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

def generate_transcript(
    audio, 
    num_speakers,
    show_all_codes,
    seed, 
    decoding, 
    cont_temperature, 
    trans_temperature,
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
        generate_kwargs["cont_temperature"] = float(cont_temperature)
        generate_kwargs["trans_temperature"] = float(trans_temperature)
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
    
    sequence = "<|audio_first|>"
    sequence += "".join([f"<|speaker|>{chr(ord('A') + i % 26)}" for i in range(num_speakers)])
    sequence += "<|end_header|>"
    sequence += "<|audio|>"

    codec_framerate = 50.0
    chunk_size_secs = 0.3
    hist_size_secs = 3.0
    sr, audio = audio
    if len(audio.shape) > 1:
        audio = audio.T
    transcribe_after = None
    start = 0
    while start < audio.shape[-1]:
        chunk_size_secs_ = hist_size_secs if start == 0 else chunk_size_secs
        chunk_size_samples = int(chunk_size_secs_*sr)
        # tokenize next audio chunk
        start_with_hist = max(0, int(start-hist_size_secs*sr))
        audio_chunk = (sr, audio[..., start:start+chunk_size_samples])
        audio_chunk_with_hist = (sr, audio[..., start_with_hist:start+chunk_size_samples])
        input_audio_str = tokenize_audio(audio_chunk_with_hist)
        actual_chunk_secs = audio_chunk[1].shape[-1] / sr
        actual_chunk_frames = int(actual_chunk_secs*codec_framerate)
        input_audio_str = input_audio_str[-actual_chunk_frames:]

        if transcribe_after is not None:
            sequence += f"{input_audio_str[:transcribe_after]}<|end_audio|>"

            # transcribe the audio
            completion = client.completions.create(
                model=model_name, 
                prompt=sequence, 
                seed=int(seed) if seed else None,
                max_tokens=100, 
                temperature=generate_kwargs["trans_temperature"],
                top_p=generate_kwargs["top_p"],
                best_of=generate_kwargs["num_beams"],
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
                max_tokens=int(chunk_size_secs*codec_framerate), 
                temperature=generate_kwargs["cont_temperature"],
                top_p=generate_kwargs["top_p"],
                best_of=generate_kwargs["num_beams"],
                extra_body=extra_body,
                stream=False,
                stop="<|end_audio|>",
            )
            if completion.choices[0].finish_reason == "stop":
                completion_text = completion.choices[0].text
                transcribe_after = len(completion_text)

        if not show_all_codes:
            output_transcript = re.sub(shorten_codes_regex, ".........", sequence)
        else:
            output_transcript = sequence

        if len(audio_chunk[1].shape) > 1:
            audio_chunk = (audio_chunk[0], audio_chunk[1].T)
        yield audio_chunk, output_transcript
        start += chunk_size_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streaming ASR demo")
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
        fn=generate_transcript,
        inputs=[
            gr.Audio(label="Audio"),
            gr.Slider(1, 10, value=2, step=1, label="Number of speakers"),
            gr.Checkbox(False, label="Show all codes"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(0.0, 10.0, value=0.6, step=0.01, label="Continuation Temperature"),
            gr.Slider(0.0, 10.0, value=0.2, step=0.01, label="Transcription Temperature"),
            gr.Slider(1, 100, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=0, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
        ], 
        outputs=[
            gr.Audio(label="Streaming Audio", streaming=True, autoplay=True),
            gr.TextArea(label="Streaming Transcript")
        ],
        allow_flagging='never'
    )
    interface.launch()