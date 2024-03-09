import argparse
import asyncio
import numpy as np
import os

from realtime_codec_agent.data_loaders.talkbank_data_loader import TalkbankDataLoader
from sklearn.model_selection import train_test_split

async def main():
    parser = argparse.ArgumentParser("Prepare the audio training dataset")
    parser.add_argument("--data-dir", default="data/audio")
    parser.add_argument("--corpora", default="All")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--test-proportion", type=float, default=0.01)
    parser.add_argument("--dev-proportion", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-num-files", type=int, default=-1)
    parser.add_argument("--encodec-model", default="facebook/encodec_24khz")
    parser.add_argument("--use-n-codebooks", type=int, default=2)
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tokenizer-offset", type=int, default=-1)
    parser.add_argument("--add-audio-tokens", action="store_true", default=False)
    
    args = parser.parse_args()
    
    loader = TalkbankDataLoader(
        args.encodec_model,
        args.tokenizer,
        use_n_codebooks=args.use_n_codebooks,
        tokenizer_offset=args.tokenizer_offset,
        add_audio_tokens=args.add_audio_tokens,
        download_dir=os.path.join(args.data_dir, "raw"), 
        force_download=args.force_download
    )
    audio_files = []
    async for audio_file, dialogue in loader.load_data(corpora=args.corpora, group_by_dialogue=True):
        dialogue_filename = os.path.join(args.data_dir, f"dialogue_{len(audio_files)}.txt")
        with open(dialogue_filename, "w", encoding="utf-8") as f:
            for example in dialogue:
                f.write(example)
                f.write("\n")
        audio_files.append(os.path.relpath(audio_file, loader.download_dir))
        if args.debug_num_files > 0 and len(audio_files) == args.debug_num_files:
            break

    audio_files = np.array(audio_files)
    
    dev_test_proportion = args.dev_proportion + args.test_proportion
    train_dialogues, test_dialogues = train_test_split(np.arange(len(audio_files)), test_size=dev_test_proportion, random_state=args.seed)
    test_proportion = args.test_proportion / dev_test_proportion
    dev_dialogues, test_dialogues = train_test_split(test_dialogues, test_size=test_proportion, random_state=args.seed)

    for split, split_dialogues in zip(("train", "dev", "test"), (train_dialogues, dev_dialogues, test_dialogues)):
        output_filename = os.path.join(args.data_dir, f"sources_{split}.txt")
        split_audio_files = sorted(audio_files[split_dialogues].tolist())
        with open(output_filename, "w", encoding="utf-8") as f:
            for audio_file in split_audio_files:
                f.write(audio_file)
                f.write("\n")

        output_filename = os.path.join(args.data_dir, f"dataset_{split}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            for dialogue in split_dialogues:
                dialogue_filename = os.path.join(args.data_dir, f"dialogue_{dialogue}.txt")
                with open(dialogue_filename, "r", encoding="utf-8") as df:
                    for example in df:
                        f.write(example)
                os.remove(dialogue_filename)
    
if __name__ == "__main__":
    asyncio.run(main())
    