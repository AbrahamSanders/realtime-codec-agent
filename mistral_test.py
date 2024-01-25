from transformers import AutoTokenizer
from realtime_codec_agent.audio_mistral import AudioMistralForCausalLM, AudioMistralConfig
import torch

from realtime_codec_agent.utils.tokenizer_utils import add_special_audio_tokens

model_name = "mistralai/Mistral-7B-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AudioMistralConfig.from_pretrained(model_name)
model = AudioMistralForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(device)

add_special_audio_tokens(tokenizer, config.num_codebooks, config.codebook_size)

inputs = tokenizer("hello <c0t0000><c0t0001><c0t0002> goodbye", return_tensors="pt").to(device)
outputs = model(**inputs)

pass

