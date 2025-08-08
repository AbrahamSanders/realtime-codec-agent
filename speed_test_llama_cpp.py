from datetime import datetime
from transformers import AutoTokenizer

from realtime_codec_agent.utils.llamacpp_utils import LlamaForAlternatingCodeChannels

llm = LlamaForAlternatingCodeChannels(
    model_path="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-Q8_0.gguf",
    n_ctx=131072,
    n_gpu_layers=-1,
    verbose=False,
    flash_attn=True,
)
tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo")

input_ids = tokenizer.encode("<|text_first|><|speaker|>A<|speaker|>B<|end_header|> A: hi how are you?<|audio|>")

start = datetime.now()

# Reset mirostat sampling
llm.init_sampler_for_generate(
    top_k=50,
    top_p=1.0,
    min_p=0.0,
    temp=1.0,
)
llm.eval(input_ids)
output_tokens = []
for _ in range(4096):
    output = next(llm.generate(input_ids, reset=False))
    output_tokens.append(output)
    input_ids = [output]
end = datetime.now()
elapsed = end - start
tokens_per_second = len(output_tokens) / elapsed.total_seconds()

output = tokenizer.decode(output_tokens, skip_special_tokens=False)

print(output)
print(f"Time taken: {elapsed.total_seconds():.2f} seconds")
print(f"Tokens generated: {len(output_tokens)}")
print(f"Tok/s: {tokens_per_second:.2f}")