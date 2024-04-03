import argparse
import asyncio
import numpy as np
import os

from realtime_codec_agent.data_loaders.audio_data_loader import AudioDataLoader
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
    
    loader = AudioDataLoader(
        args.encodec_model,
        args.tokenizer,
        use_n_codebooks=args.use_n_codebooks,
        tokenizer_offset=args.tokenizer_offset,
        add_audio_tokens=args.add_audio_tokens,
        download_dir=os.path.join(args.data_dir, "raw"), 
        force_download=args.force_download
    )
    num_files = 0
    async for audio_file, dialogue in loader.load_data(corpora=args.corpora, group_by_dialogue=True):
        dialogue_filename = os.path.join(args.data_dir, f"dialogue_{num_files}.txt")
        with open(dialogue_filename, "w", encoding="utf-8") as f:
            f.write(os.path.relpath(audio_file, loader.download_dir))
            f.write("\n")
            for example in dialogue:
                f.write(example)
                f.write("\n")
        num_files += 1
        if args.debug_num_files > 0 and num_files == args.debug_num_files:
            break
    
    dev_test_proportion = args.dev_proportion + args.test_proportion
    train_dialogues, test_dialogues = train_test_split(np.arange(num_files), test_size=dev_test_proportion, random_state=args.seed)
    test_proportion = args.test_proportion / dev_test_proportion
    dev_dialogues, test_dialogues = train_test_split(test_dialogues, test_size=test_proportion, random_state=args.seed)

    train_dialogues.sort()
    dev_dialogues.sort()
    test_dialogues.sort()

    for split, split_dialogues in zip(("train", "dev", "test"), (train_dialogues, dev_dialogues, test_dialogues)):
        sources_output_filename = os.path.join(args.data_dir, f"sources_{split}.txt")
        dataset_output_filename = os.path.join(args.data_dir, f"dataset_{split}.txt")
        with open(sources_output_filename, "w", encoding="utf-8") as f_sources:
            with open(dataset_output_filename, "w", encoding="utf-8") as f_dataset:
                for dialogue in split_dialogues:
                    dialogue_filename = os.path.join(args.data_dir, f"dialogue_{dialogue}.txt")
                    with open(dialogue_filename, "r", encoding="utf-8") as f_dialogue:
                        for i, example in enumerate(f_dialogue):
                            if i == 0:
                                f_sources.write(example)
                            else:
                                f_dataset.write(example)
                    os.remove(dialogue_filename)
    
if __name__ == "__main__":
    asyncio.run(main())
    