from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("data/audio/output/Llama-3.2-3B-magicodec-no-bpe-131k")

# open data/audio/dataset_train.txt, tokenize each line, and compute the sequence length of each line.
lengths = set()
with open("data/audio/dataset_multi_no_bpe_magicodec_stereo_131k_8192.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tokens = tokenizer(line.rstrip("\n")).input_ids
        lengths.add(len(tokens))

# assert that they are all the same
print(f"distinct lengths: {lengths}")