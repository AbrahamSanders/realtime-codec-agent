from transformers import (
    #AutoModelForCausalLM, 
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

from realtime_codec_agent.audio_mistral import AudioMistralForCausalLM
from transformers.generation import AlternatingCodebooksLogitsProcessor
from realtime_codec_agent.utils.encodec_utils import n_codebooks_to_bandwidth_id

tokenizer = None
device = None
model = None

encodec_model = None
encodec_processor = None
vocos = None

def tokenize_audio(audio, use_n_codebooks, tokenizer_offset):
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
    bandwidth_id = n_codebooks_to_bandwidth_id(use_n_codebooks)
    bandwidth = encodec_model.config.target_bandwidths[bandwidth_id]
    encoder_outputs = encodec_model.encode(**inputs, bandwidth=bandwidth).audio_codes # 1 x 1 x n_codebooks x n_tokens

    # convert to sequence of alternating tokens
    audio_codes = torch.zeros(encoder_outputs.shape[-1] * use_n_codebooks + 1, dtype=encoder_outputs.dtype).to(device)
    audio_codes[0] = tokenizer.bos_token_id
    for i in range(use_n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[1:][i::use_n_codebooks] = tokenizer_offset + codebook_offset + encoder_outputs[0, 0, i]

    # return as model inputs
    audio_codes = audio_codes.unsqueeze(0)
    attention_mask = torch.ones_like(audio_codes)
    return {"input_ids": audio_codes, "attention_mask": attention_mask}

def detokenize_audio(tokens, n_codebooks, tokenizer_offset):
    # convert sequence of alternating tokens to audio codes
    audio_codes = torch.zeros((n_codebooks, (tokens.shape[-1] - 1) // n_codebooks), dtype=tokens.dtype).to(device)
    for i in range(n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[i] = tokens[0, 1:][i::n_codebooks] - codebook_offset - tokenizer_offset
    
    # decode audio codes with encodec
    with torch.no_grad():
        decoder_outputs = encodec_model.decode(audio_codes.unsqueeze(0).unsqueeze(0), [None])
        output_audio_encodec = decoder_outputs.audio_values[0, 0]

    # decode audio codes with vocos
    bandwidth_id = n_codebooks_to_bandwidth_id(n_codebooks)
    bandwidth_id = torch.tensor([bandwidth_id]).to(device)
    features = vocos.codes_to_features(audio_codes)
    output_audio_vocos = vocos.decode(features, bandwidth_id=bandwidth_id)[0]

    # return audio
    sr = encodec_model.config.sampling_rate
    return (sr, output_audio_encodec.cpu().numpy()), (sr, output_audio_vocos.cpu().numpy())

def generate_audio(context_audio, seed, decoding, seconds, temperature, num_beams, top_k, top_p, typical_p, penalty_alpha):
    use_n_codebooks = 2
    tokenizer_offset = len(tokenizer) - use_n_codebooks * encodec_model.config.codebook_size

    generate_kwargs = {
        "num_beams": int(num_beams),
        "early_stopping": True,
        "max_new_tokens": int(seconds) * use_n_codebooks * 75,
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

    model_inputs = tokenize_audio(context_audio, use_n_codebooks, tokenizer_offset)
    altCodebooksLogitsProc = AlternatingCodebooksLogitsProcessor(
        input_start_len = model_inputs["input_ids"].shape[-1],
        semantic_vocab_size = tokenizer_offset,
        codebook_size = encodec_model.config.codebook_size
    )
    model_outputs = model.generate(**model_inputs, **generate_kwargs, logits_processor=[altCodebooksLogitsProc])
    output_audio_encodec, output_audio_vocos = detokenize_audio(model_outputs, use_n_codebooks, tokenizer_offset)

    return output_audio_encodec, output_audio_vocos

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the audio generator demo")
    parser.add_argument("--model", type=str, default="Mistral-7B-realtime-codec-agent-2/checkpoint-1824")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioMistralForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)

    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

    interface = gr.Interface(
        fn=generate_audio,
        inputs=[
            gr.Audio(label="Context"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(1, 20, value=2, step=1, label="Seconds"),
            gr.Slider(0.1, 10.0, value=1.0, step=0.01, label="Temperature"),
            gr.Slider(1, 10, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=0, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
            gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Penalty-alpha")
        ], 
        outputs=[
            gr.Audio(label="Continuation (EnCodec decoder)"),
            gr.Audio(label="Continuation (Vocos decoder)"),
        ],
        allow_flagging='never'
    )
    interface.launch()