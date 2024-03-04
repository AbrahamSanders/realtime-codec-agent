from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from realtime_codec_agent.utils.tokenizer_utils import add_special_audio_tokens

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
add_special_audio_tokens(tokenizer, config.vocab_size, 2, 1024)

# open data/audio/dataset_train.txt, tokenize each line, and compute the sequence length of each line.
lengths = set()
min_token = len(tokenizer)
with open("data/audio/dataset_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        tokens = tokenizer(line.rstrip("\n")).input_ids
        lengths.add(len(tokens))
        bos_offset = 1 if tokenizer.bos_token_id is not None else 0
        line_min_token = min(tokens[bos_offset:])
        if line_min_token < min_token:
            min_token = line_min_token

# assert that they are all the same
print(f"distinct lengths: {lengths}")
# print the minimum token
print(f"min token (excluding bos): {tokenizer.convert_ids_to_tokens(min_token)} ({min_token})")