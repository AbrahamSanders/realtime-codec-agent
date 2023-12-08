from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("adept/persimmon-8b-base")

# open data/audio/dataset_train.txt, tokenize each line, and compute the sequence length of each line.
lengths = []
with open("data/audio/dataset_dev_debug.txt", "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tokens = tokenizer(line).input_ids
        lengths.append(len(tokens))

# assert that they are all the same
lengths_set = set(lengths)
print(lengths_set)