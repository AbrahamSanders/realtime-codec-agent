import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a codec BPE dataset into audio-first and text-first datasets")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    audio_first_path = args.dataset_path.replace(".txt", "_audio_first.txt")
    text_first_path = args.dataset_path.replace(".txt", "_text_first.txt")
    with open(args.dataset_path, encoding="utf-8") as f_source:
        with open(audio_first_path, "w", encoding="utf-8") as f_audio_first:
            with open(text_first_path, "w", encoding="utf-8") as f_text_first:
                for line in tqdm(f_source, desc="Examples"):
                    if line.startswith("<|audio_first|>"):
                        f_audio_first.write(line)
                    elif line.startswith("<|text_first|>"):
                        f_text_first.write(line)
                    else:
                        raise ValueError(f"Invalid line format: {line}")
    print("Done!")