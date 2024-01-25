import argparse
import asyncio
from os import path
from itertools import chain

from realtime_codec_agent.data_loaders.talkbank_data_loader import TalkbankDataLoader
from sklearn.model_selection import train_test_split

async def main():
    parser = argparse.ArgumentParser("Prepare the audio training dataset")
    parser.add_argument("--data-dir", default="data/audio")
    parser.add_argument("--corpora", default="All")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--test-proportion", type=float, default=0.1)
    parser.add_argument("--dev-proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
        download_dir=path.join(args.data_dir, "raw"), 
        force_download=args.force_download
    )
    dialogues = []
    async for dialogue in loader.load_data(corpora=args.corpora, group_by_dialogue=True):
        dialogues.append(dialogue)
    
    train_dialogues, test_dialogues = train_test_split(dialogues, test_size=args.test_proportion, random_state=args.seed)
    train_dialogues, dev_dialogues = train_test_split(train_dialogues, test_size=args.dev_proportion, random_state=args.seed)
    train_examples = list(chain(*train_dialogues))
    dev_examples = list(chain(*dev_dialogues))
    test_examples = list(chain(*test_dialogues))

    for split, split_examples in zip(("train", "dev", "test"), (train_examples, dev_examples, test_examples)):
        output_filename = path.join(args.data_dir, f"dataset_{split}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            for example in split_examples:
                f.write(example)
                f.write("\n")
    
if __name__ == "__main__":
    asyncio.run(main())
    