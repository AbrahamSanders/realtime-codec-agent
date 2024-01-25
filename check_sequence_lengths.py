from transformers import AutoTokenizer
from tqdm import tqdm

from realtime_codec_agent.utils.tokenizer_utils import add_special_audio_tokens

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
add_special_audio_tokens(tokenizer, 2, 1024)

# open data/audio/dataset_train.txt, tokenize each line, and compute the sequence length of each line.
lengths = []
with open("data/audio/dataset_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tokens = tokenizer(line.rstrip("\n")).input_ids
        lengths.append(len(tokens))

# assert that they are all the same
lengths_set = set(lengths)
print(lengths_set)