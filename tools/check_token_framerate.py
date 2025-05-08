from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute token framerate for a plain-text codec BPE dataset")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, required=True)
    parser.add_argument("--codec_framerate", type=float, required=True)
    parser.add_argument("--vocab_cutoff_token", type=str, default="<|end_header|>")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_cutoff = tokenizer.convert_tokens_to_ids(args.vocab_cutoff_token)
    token_framerates = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            if args.num_examples and i == args.num_examples:
                break
            tokens = tokenizer(line.rstrip("\n"), return_tensors="np").input_ids
            audio_tokens = tokens[tokens > vocab_cutoff]
            audio_str = tokenizer.decode(audio_tokens)
            num_units = len(audio_str) / args.num_codebooks
            num_secs = num_units / args.codec_framerate
            token_framerate = audio_tokens.shape[-1] / num_secs
            token_framerates.append(token_framerate)

    print(f"Max: {np.max(token_framerates):.2f} tokens/second")
    print(f"Min: {np.min(token_framerates):.2f} tokens/second")
    print(f"Median: {np.median(token_framerates):.2f} tokens/second")
    print(f"Mean: {np.mean(token_framerates):.2f} tokens/second")
    print(f"Std: {np.std(token_framerates):.2f} tokens/second")