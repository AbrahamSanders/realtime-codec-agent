from transformers import AutoTokenizer, EncodecModel
from realtime_codec_agent.audio_mistral import AudioMistralForCausalLM, AudioMistralConfig
#from realtime_codec_agent.audio_qwen2 import AudioQwen2ForCausalLM, AudioQwen2Config
import torch

from realtime_codec_agent.utils.tokenizer_utils import add_special_audio_tokens

model_name = "mistralai/Mistral-7B-v0.1"
#model_name = "Qwen/Qwen1.5-1.8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = AudioMistralConfig.from_pretrained(model_name)
#config = AudioQwen2Config.from_pretrained(model_name)
model = AudioMistralForCausalLM.from_pretrained(model_name, config=config).to(device)
#model = AudioQwen2ForCausalLM.from_pretrained(model_name, config=config).to(device)

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
with torch.no_grad():
    cb_size = encodec_model.config.codebook_size
    for i in range(config.num_codebooks):
        model.model.audio_embed[i*cb_size:(i+1)*cb_size] = encodec_model.quantizer.layers[i].codebook.embed.clone()

add_special_audio_tokens(tokenizer, config.vocab_size, config.num_codebooks, config.codebook_size)

inputs = tokenizer(["hello <c0t0000><c1t0000><c0t0001><c1t0001> goodbye",
                    "hello <c0t1022><c1t1022><c0t1023><c1t1023> goodbye"], return_tensors="pt", padding=True).to(device)
outputs = model(**inputs, labels=inputs.input_ids)

pass

