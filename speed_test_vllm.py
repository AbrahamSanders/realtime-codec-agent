from typing import List, Union
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
from tqdm import trange
import torch
import time
import librosa
import numpy as np
import soundfile as sf

from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim

DEBUG = False

class GenerateParams:
    def __init__(self, input_ids: torch.LongTensor, sampling_params: SamplingParams):
        self.input_ids = input_ids
        self.sampling_params = sampling_params

def trim_sequences(
    audio_first_input_ids, text_first_input_ids, audio_first_max_seq_length, text_first_max_seq_length, audio_first_trim_pos, 
    text_first_trim_pos, text_first_trans_pos, audio_first_trim_by, text_first_trim_by
):
    if audio_first_input_ids.shape[-1] >= audio_first_max_seq_length:
        audio_first_input_ids = torch.cat(
            [
                audio_first_input_ids[..., :audio_first_trim_pos],
                audio_first_input_ids[..., audio_first_trim_pos+audio_first_trim_by:],
            ], 
            dim=1,
        )
    if text_first_input_ids.shape[-1] >= text_first_max_seq_length:
        text_first_input_ids = torch.cat(
            [
                text_first_input_ids[..., :text_first_trim_pos],
                text_first_input_ids[..., text_first_trim_pos+text_first_trim_by:],
            ], 
            dim=1,
        )
        text_first_trans_pos = max(text_first_trim_pos, text_first_trans_pos-text_first_trim_by)
    return audio_first_input_ids, text_first_input_ids, text_first_trans_pos

def generate(llm: LLM, generate_params: Union[List[GenerateParams], GenerateParams]) -> Union[List[torch.LongTensor], torch.LongTensor]:
    if isinstance(generate_params, GenerateParams):
        generate_params = [generate_params]
    prompts = [TokensPrompt(prompt_token_ids=params.input_ids[0].tolist()) for params in generate_params]
    sampling_params = [params.sampling_params for params in generate_params]
    for params in sampling_params:
        params.skip_special_tokens = False
        params.spaces_between_special_tokens = False
        #params.seed = 42
        params.detokenize = bool(params.stop)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
    next_token_ids = [torch.tensor(output.outputs[0].token_ids).unsqueeze(0) for output in outputs]
    if len(next_token_ids) == 1:
        return next_token_ids[0]
    return next_token_ids

def generate_for_mode(llm, audio_first_input_ids, text_first_input_ids, mode, start_audio_token_id, end_audio_token_id, force_trans):
    if mode == "audio" or mode == "force_audio":
        logit_bias = {end_audio_token_id: -100} if mode == "force_audio" else None
        audio_first_next_tokens = torch.LongTensor([[end_audio_token_id if force_trans else start_audio_token_id]])
        text_first_next_tokens = generate(
            llm, 
            [
                GenerateParams(
                    text_first_input_ids, 
                    SamplingParams(max_tokens=1, temperature=1.0, min_p=0.002, logit_bias=logit_bias),
                ),
            ]
        )
        generated_tokens = 1
    elif mode == "both_text":
        audio_first_next_tokens, text_first_next_tokens = generate(
            llm, 
            [
                GenerateParams(
                    audio_first_input_ids, 
                    SamplingParams(max_tokens=100, temperature=0.2, stop_token_ids=[start_audio_token_id]),
                ),
                GenerateParams(
                    text_first_input_ids, 
                    SamplingParams(max_tokens=100, temperature=1.0, min_p=0.002, stop_token_ids=[start_audio_token_id], stop=" B:"),
                ),
            ]
        )
        generated_tokens = audio_first_next_tokens.shape[-1] + text_first_next_tokens.shape[-1]
    elif mode == "audio_first_text":
        audio_first_next_tokens = generate(
            llm,
            GenerateParams(
                audio_first_input_ids, 
                SamplingParams(max_tokens=100, temperature=0.2, stop_token_ids=[start_audio_token_id]),
            ),
        )
        generated_tokens = audio_first_next_tokens.shape[-1]
        text_first_next_tokens = None
    elif mode == "text_first_text":
        text_first_next_tokens = generate(
            llm,
            GenerateParams(
                text_first_input_ids, 
                SamplingParams(max_tokens=100, temperature=1.0, min_p=0.002, stop_token_ids=[start_audio_token_id], stop=" B:"),
            ),
        )
        generated_tokens = text_first_next_tokens.shape[-1]
        audio_first_next_tokens = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if "text" in mode:
        if audio_first_next_tokens is not None and audio_first_next_tokens[0, -1] != start_audio_token_id:
            audio_first_next_tokens = torch.cat([audio_first_next_tokens, torch.LongTensor([[start_audio_token_id]])], dim=1)
        if text_first_next_tokens is not None and text_first_next_tokens[0, -1] != start_audio_token_id:
            text_first_next_tokens = torch.cat([text_first_next_tokens, torch.LongTensor([[start_audio_token_id]])], dim=1)

    return audio_first_next_tokens, text_first_next_tokens, generated_tokens

if __name__ == "__main__":
    model_name = "Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=2, 
        gpu_memory_utilization=0.7, 
        enable_prefix_caching=True, 
        enforce_eager=DEBUG,
        # compilation_config={
        #     "full_cuda_graph": True,
        # },
    )

    audio_tokenizer = AudioTokenizer()

    audio_file = "experimental/audio (4) (1).wav"
    audio, _ = librosa.load(audio_file, sr=audio_tokenizer.sampling_rate, mono=False)
    audio = audio[0]

    chunk_size_secs = 0.1
    if int(chunk_size_secs*100) % 2 != 0:
        raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
    chunk_size_samples = int(chunk_size_secs * audio_tokenizer.sampling_rate)
    audio_first_max_seq_length = 4096
    audio_first_trim_by = 1024
    text_first_max_seq_length = 4096
    text_first_trim_by = 1024
    crossfade_ramps = create_crossfade_ramps(audio_tokenizer.sampling_rate, fade_secs=0.02)

    audio_first_prompt = "<|audio_first|><|speaker|>A<|speaker|>B<|end_header|><|audio|>"
    audio_first_input_ids = tokenizer(audio_first_prompt, return_tensors="pt").input_ids
    text_first_prompt = "<|text_first|><|speaker|>A<|speaker|>B<|end_header|><|audio|>"
    text_first_input_ids = tokenizer(text_first_prompt, return_tensors="pt").input_ids
    audio_first_trim_pos = audio_first_input_ids.shape[-1]-1
    text_first_trim_pos = text_first_trans_pos = text_first_input_ids.shape[-1]-1

    generated_tokens = 0
    generated_audio_tokens = 0
    out_audio_ch1, out_audio_ch2 = np.zeros((2, 0), dtype=np.float32)
    
    end_header_token_id = tokenizer.convert_tokens_to_ids("<|end_header|>")
    start_audio_token_id = tokenizer.convert_tokens_to_ids("<|audio|>")
    end_audio_token_id = tokenizer.convert_tokens_to_ids("<|end_audio|>")
    user_speaker_token_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    start_time = time.time()
    last_in_chunk_abs_max = 0.0
    last_out_chunk_abs_max = 0.0
    out_chunk = None
    trans_abs_max_threshold = 100 / 32768.0
    for start in trange(0, len(audio), chunk_size_samples, desc="Chunks"):
        end = start + chunk_size_samples
        audio_chunk = audio[start:end]
        in_chunk_abs_max = np.abs(audio_chunk).max()
        out_chunk_abs_max = 0.0 if out_chunk is None else np.abs(out_chunk).max()
        force_trans = (
            (last_in_chunk_abs_max >= trans_abs_max_threshold and in_chunk_abs_max < trans_abs_max_threshold) or
            (last_out_chunk_abs_max >= trans_abs_max_threshold and out_chunk_abs_max < trans_abs_max_threshold)
        )
        last_in_chunk_abs_max = in_chunk_abs_max
        last_out_chunk_abs_max = out_chunk_abs_max
        audio_chunk_str = audio_tokenizer.tokenize_audio(audio_chunk)
        audio_chunk_input_ids = tokenizer(audio_chunk_str, add_special_tokens=False, return_tensors="pt").input_ids
        out_chunk_input_ids = torch.zeros((1, 0), dtype=audio_chunk_input_ids.dtype)
        for i in range(audio_chunk_input_ids.shape[-1]):
            mode = "audio"
            while True:
                # trim the sequences to the maximum length
                audio_first_input_ids, text_first_input_ids, text_first_trans_pos = trim_sequences(
                    audio_first_input_ids, text_first_input_ids, audio_first_max_seq_length, text_first_max_seq_length, 
                    audio_first_trim_pos, text_first_trim_pos, text_first_trans_pos, audio_first_trim_by, text_first_trim_by
                )
                # predict next tokens
                audio_first_next_tokens, text_first_next_tokens, generated_tokens_ = generate_for_mode(
                    llm, audio_first_input_ids, text_first_input_ids, mode, start_audio_token_id, end_audio_token_id, force_trans
                )
                force_trans = False
                generated_tokens += generated_tokens_
                if mode == "audio" or mode == "force_audio":
                    if audio_first_next_tokens[0, 0] == end_audio_token_id and text_first_next_tokens[0, 0] == end_audio_token_id:
                        mode = "both_text"
                    elif audio_first_next_tokens[0, 0] == end_audio_token_id:
                        mode = "audio_first_text"
                    elif text_first_next_tokens[0, 0] == end_audio_token_id:
                        mode = "text_first_text"
                    else:
                        next_input_token = audio_chunk_input_ids[..., i:i+1]
                        audio_first_input_ids = torch.cat([audio_first_input_ids, text_first_next_tokens, next_input_token], dim=1)
                        text_first_input_ids = torch.cat([text_first_input_ids, text_first_next_tokens, next_input_token], dim=1)
                        out_chunk_input_ids = torch.cat([out_chunk_input_ids, text_first_next_tokens], dim=1)
                        generated_audio_tokens += 1
                        mode = "audio"
                        break # move on to next input token
                if mode == "audio_first_text" or mode == "both_text":
                    audio_first_input_ids = torch.cat([audio_first_input_ids, audio_first_next_tokens], dim=1)
                    if audio_first_next_tokens[0, 0] != end_audio_token_id:
                        if mode != "both_text":
                            mode = "audio"
                        # splice the transcription into the text-first sequence if it belongs to the user speaker
                        if audio_first_next_tokens[0, 0] == user_speaker_token_id:
                            add_control_tokens = text_first_input_ids[0, text_first_trans_pos] != start_audio_token_id
                            audio_start_id = torch.LongTensor([[start_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
                            audio_end_id = torch.LongTensor([[end_audio_token_id]]) if add_control_tokens else torch.LongTensor([[]])
                            text_first_input_ids = torch.cat(
                                [
                                    text_first_input_ids[..., :text_first_trans_pos],
                                    audio_end_id, 
                                    audio_first_next_tokens[..., :-1],
                                    audio_start_id,
                                    text_first_input_ids[..., text_first_trans_pos:]
                                ], 
                                dim=1,
                            )
                        text_first_trans_pos = text_first_input_ids.shape[-1]-1 \
                            if text_first_input_ids[0, -1] == start_audio_token_id else text_first_input_ids.shape[-1]
                if mode == "text_first_text" or mode == "both_text":
                    if text_first_next_tokens[0, 0] == user_speaker_token_id:
                        # discard predictions for the user speaker on the text-first sequence
                        # and force the next token to be an audio token
                        text_first_input_ids = text_first_input_ids[..., :-1]
                        mode = "force_audio"
                    else:
                        text_first_input_ids = torch.cat([text_first_input_ids, text_first_next_tokens], dim=1)
                        if text_first_next_tokens[0, 0] != end_audio_token_id:
                            mode = "audio"

        out_chunk_str = tokenizer.decode(out_chunk_input_ids[0], skip_special_tokens=False)
        (_, out_chunk), _, preroll_samples = audio_tokenizer.detokenize_audio(out_chunk_str, preroll_samples=crossfade_ramps[0])
        out_chunk = pad_or_trim(out_chunk, audio_chunk.shape[-1] + preroll_samples)
        out_audio_ch1 = smooth_join(out_audio_ch1, out_chunk, *crossfade_ramps)
        out_audio_ch2 = np.concatenate((out_audio_ch2, audio_chunk), axis=-1)

    out_audio = np.stack([out_audio_ch1, out_audio_ch2], axis=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("---------------------------------------------------------------------------------------")
    print("-- Audio First Sequence:")
    print("---------------------------------------------------------------------------------------")
    audio_first_output_str = tokenizer.decode(audio_first_input_ids[0], skip_special_tokens=False)
    print(audio_first_output_str)
    
    print("---------------------------------------------------------------------------------------")
    print("-- Text First Sequence:")
    print("---------------------------------------------------------------------------------------")
    text_first_output_str = tokenizer.decode(text_first_input_ids[0], skip_special_tokens=False)
    print(text_first_output_str)

    print("---------------------------------------------------------------------------------------")
    print("-- Metrics:")
    print("---------------------------------------------------------------------------------------")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Generated Tokens (total): {generated_tokens}")
    tokens_per_second = generated_tokens / elapsed_time
    print(f"Tokens per second (total): {tokens_per_second:.2f}")
    print(f"Generated Tokens (audio): {generated_audio_tokens}")
    audio_tokens_per_second = generated_audio_tokens / elapsed_time
    print(f"Tokens per second (audio): {audio_tokens_per_second:.2f}")
    print(f"Audio-First Sequence Length: {audio_first_input_ids.shape[-1]}")
    print(f"Text-First Sequence Length: {text_first_input_ids.shape[-1]}")
    # Save the output audio
    output_file = "output.wav"
    sf.write(output_file, out_audio.T, audio_tokenizer.sampling_rate)
    