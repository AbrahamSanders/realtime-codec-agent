from transformers import (
    #AutoModelForCausalLM, 
    AutoTokenizer, 
    EncodecModel, 
    AutoProcessor,
    SinkCache
)
from transformers.trainer_utils import set_seed
import torch
import gradio as gr
import argparse
import librosa
from vocos import Vocos

from realtime_codec_agent.audio_mistral import AudioMistralForCausalLM
from realtime_codec_agent.audio_qwen2 import AudioQwen2ForCausalLM
from transformers.generation import AlternatingCodebooksLogitsProcessor
from realtime_codec_agent.utils.generate_utils import ExcludeCodebooksLogitsProcessor
from realtime_codec_agent.utils.encodec_utils import n_codebooks_to_bandwidth_id

tokenizer = None
device = None
model = None

encodec_model = None
encodec_processor = None
vocos = None

def get_model_class(model_name):
    model_name = model_name.lower()
    if "mistral" in model_name:
        return AudioMistralForCausalLM
    elif "qwen" in model_name:
        return AudioQwen2ForCausalLM
    else:
        raise ValueError(f"Model type for {model_name} not recognized")

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
    bos_offset = 1 if tokenizer.bos_token_id is not None else 0
    audio_codes = torch.zeros(encoder_outputs.shape[-1] * use_n_codebooks + bos_offset, dtype=encoder_outputs.dtype).to(device)
    if bos_offset:
        audio_codes[0] = tokenizer.bos_token_id
    for i in range(use_n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[bos_offset:][i::use_n_codebooks] = tokenizer_offset + codebook_offset + encoder_outputs[0, 0, i]

    # return as model inputs
    audio_codes = audio_codes.unsqueeze(0)
    attention_mask = torch.ones_like(audio_codes)
    return {"input_ids": audio_codes, "attention_mask": attention_mask}

def detokenize_audio(tokens, n_codebooks, tokenizer_offset):
    # convert sequence of alternating tokens to audio codes
    bos_offset = 1 if tokenizer.bos_token_id is not None else 0
    audio_codes = torch.zeros((n_codebooks, (tokens.shape[-1] - bos_offset) // n_codebooks), dtype=tokens.dtype).to(device)
    for i in range(n_codebooks):
        codebook_offset = i * encodec_model.config.codebook_size
        audio_codes[i] = tokens[0, bos_offset:][i::n_codebooks] - codebook_offset - tokenizer_offset
    
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

def prepare_sink_cache(num_sink_tokens, window_length_seconds, use_n_codebooks, generate_kwargs):
    window_length = int(window_length_seconds) * use_n_codebooks * 75
    print (f"Using SinkCache with window_length={window_length} and num_sink_tokens={num_sink_tokens}.")
    cache = SinkCache(window_length=window_length, num_sink_tokens=num_sink_tokens)
    generate_kwargs["past_key_values"] = cache
    generate_kwargs["use_cache"] = True

def generate_audio(
        context_audio, 
        seed, 
        generate_audio,
        use_sink_cache, 
        num_sink_tokens, 
        window_length_seconds, 
        decoding, 
        seconds, 
        temperature, 
        num_beams, 
        top_k, 
        top_p, 
        typical_p, 
        penalty_alpha
    ):
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

    model_inputs = tokenize_audio(context_audio, use_n_codebooks, tokenizer_offset)

    # Generate audio continuation
    output_audio_encodec, output_audio_vocos = None, None
    if generate_audio:
        altCodebooksLogitsProc = AlternatingCodebooksLogitsProcessor(
            input_start_len = model_inputs["input_ids"].shape[-1],
            semantic_vocab_size = tokenizer_offset,
            codebook_size = encodec_model.config.codebook_size
        )
        if use_sink_cache:
            prepare_sink_cache(num_sink_tokens, window_length_seconds, use_n_codebooks, generate_kwargs)
        if seed:
            set_seed(int(seed))
        model_outputs = model.generate(**model_inputs, **generate_kwargs, logits_processor=[altCodebooksLogitsProc])
        if use_sink_cache:
            cache = generate_kwargs["past_key_values"]
            print(f"key_cache: {cache.key_cache[0].shape}\nvalue_cache: {cache.value_cache[0].shape};\nseen_tokens: {cache.seen_tokens}")
        output_audio_encodec, output_audio_vocos = detokenize_audio(model_outputs, use_n_codebooks, tokenizer_offset)

    # Generate text continuation
    excCodebooksLogitsProc = ExcludeCodebooksLogitsProcessor(semantic_vocab_size=tokenizer_offset)
    if use_sink_cache:
        prepare_sink_cache(num_sink_tokens, window_length_seconds, use_n_codebooks, generate_kwargs)
    # rough approximation: 2.5 spoken words per second, 1.5 tokens per word
    generate_kwargs["max_new_tokens"] = round(int(seconds) * 2.5 * 1.5)
    if seed:
        set_seed(int(seed))
    model_outputs = model.generate(**model_inputs, **generate_kwargs, logits_processor=[excCodebooksLogitsProc])
    text_start = model_inputs["input_ids"].shape[-1]
    output_text = tokenizer.decode(model_outputs[0, text_start:], skip_special_tokens=False)

    return output_audio_encodec, output_audio_vocos, output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the audio generator demo")
    parser.add_argument("--model", type=str, default="Qwen1.5-1.8B-realtime-codec-agent-aligned-mini-stage-2/checkpoint-7812")
    args = parser.parse_args()

    print(f"Running with args: {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = get_model_class(args.model)
    model = model_cls.from_pretrained(args.model, torch_dtype=torch.float16).to(device)

    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)

    interface = gr.Interface(
        fn=generate_audio,
        inputs=[
            gr.Audio(label="Context"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Checkbox(True, label="Generate Audio"),
            gr.Checkbox(False, label="Use SinkCache"),
            gr.Slider(1, 32, value=8, step=1, label="Num Sink Tokens"),
            gr.Slider(1, 60, value=20, step=1, label="Window Length (seconds)"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Sampling", label="Decoding"),
            gr.Slider(1, 60, value=10, step=1, label="Seconds"),
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
            gr.TextArea(label="Continuation (Text)")
        ],
        allow_flagging='never'
    )
    interface.launch()