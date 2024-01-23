from transformers import AutoTokenizer
from audio_mistral import AudioMistralForCausalLM
import torch

model_name = "mistralai/Mistral-7B-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AudioMistralForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

tokenizer.add_tokens([f"aud{i}" for i in range(2048)], special_tokens=True)

inputs = tokenizer("hello aud0aud1aud2 goodbye").to(device)
outputs = model(**inputs)

pass

