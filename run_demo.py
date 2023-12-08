from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    EncodecModel, 
    AutoProcessor
)
from transformers.trainer_utils import set_seed
import torch
import gradio as gr
import argparse
import librosa
from vocos import Vocos

from realtime_codec_agent.utils.generate_utils import AlternatingCodebooksLogitsProcessor

tokenizer = None
device = None
model = None

encodec_model = None
encodec_processor = None
vocos = None

def n_codebooks_to_bandwidth_id(n_codebooks):
    if n_codebooks <= 2:
        bandwidth_id = 0
    elif n_codebooks <= 4:
        bandwidth_id = 1
    elif n_codebooks <= 8:
        bandwidth_id = 2
    else:
        bandwidth_id = 3
    return bandwidth_id

def tokenize_audio(audio, n_codebooks, tokenizer_offset=0):
    orig_sr, audio = audio
    audio = audio.astype("float32") / 32768.0
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.T)
    sr = encodec_model.config.sampling_rate
    # resample to encodec's sample rate if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # get encodec audio codes
    inputs = encodec_processor(raw_audio=audio, sampling_rate=sr, return_tensors="pt").to(device)
    bandwidth_id = n_codebooks_to_bandwidth_id(n_codebooks)
    bandwidth = encodec_model.config.target_bandwidths[bandwidth_id]
    encoder_outputs = encodec_model.encode(**inputs, bandwidth=bandwidth).audio_codes # 1 x 1 x n_codebooks x n_tokens

    # convert to sequence of alternating tokens
    n_codebooks = encoder_outputs.shape[-2]
    audio_codes = torch.zeros(encoder_outputs.shape[-1] * n_codebooks + 1, dtype=encoder_outputs.dtype).to(device)
    audio_codes[0] = tokenizer.bos_token_id
    for i in range(n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[1:][i::n_codebooks] = tokenizer_offset + codebook_offset + encoder_outputs[0, 0, i]

    # return as model inputs
    audio_codes = audio_codes.unsqueeze(0)
    attention_mask = torch.ones_like(audio_codes)
    return {"input_ids": audio_codes, "attention_mask": attention_mask}

def detokenize_audio(tokens, n_codebooks, tokenizer_offset=0):
    # convert sequence of alternating tokens to audio codes
    audio_codes = torch.zeros((n_codebooks, (tokens.shape[-1] - 1) // n_codebooks), dtype=tokens.dtype).to(device)
    for i in range(n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[i] = tokens[0, 1:][i::n_codebooks] - codebook_offset - tokenizer_offset
    
    # decode audio codes with vocos
    bandwidth_id = n_codebooks_to_bandwidth_id(n_codebooks)
    bandwidth_id = torch.tensor([bandwidth_id]).to(device)
    features = vocos.codes_to_features(audio_codes)
    output_audio = vocos.decode(features, bandwidth_id=bandwidth_id)[0]

    # return audio
    sr = encodec_model.config.sampling_rate
    return (sr, output_audio.cpu().numpy())

def generate_audio(context_audio_upload, context_audio_record, seed, decoding, seconds, temperature, num_beams, 
                    top_k, top_p, typical_p, penalty_alpha):
    n_codebooks = 2

    generate_kwargs = {
        "num_beams": int(num_beams),
        "early_stopping": True,
        "max_new_tokens": int(seconds) * n_codebooks * 75,
        "do_sample": False
    }
    if decoding != "Greedy":
        generate_kwargs["top_k"] = int(top_k)
    if decoding == "Sampling":
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)
        generate_kwargs["typical_p"] = float(typical_p)
    if decoding == "Contrastive":
        generate_kwargs["penalty_alpha"] = float(penalty_alpha)

    if seed:
        set_seed(int(seed))

    context_audio = context_audio_upload if context_audio_upload is not None else context_audio_record
    model_inputs = tokenize_audio(context_audio, n_codebooks)
    altCodebooksLogitsProc = AlternatingCodebooksLogitsProcessor(
        input_start_len = model_inputs["input_ids"].shape[-1],
        semantic_vocab_size = 0,
        codebook_size = encodec_model.config.codebook_size
    )
    model_outputs = model.generate(**model_inputs, **generate_kwargs, logits_processor=[altCodebooksLogitsProc])
    output_audio = detokenize_audio(model_outputs, n_codebooks)

    return output_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the audio generator demo")
    parser.add_argument("--model", type=str, default="persimmon-8b-realtime-codec-agent/checkpoint-912")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)

    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

    interface = gr.Interface(
        fn=generate_audio,
        inputs=[
            gr.Audio(label="Context (Upload)"),
            gr.Audio(label="Context (Record)", source="microphone"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(1, 20, value=2, step=1, label="Seconds"),
            gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Temperature"),
            gr.Slider(1, 10, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=0, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
            gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Penalty-alpha")
        ], 
        outputs=[
            gr.Audio(label="Continuation")
        ],
        allow_flagging='never'
    )
    interface.launch()