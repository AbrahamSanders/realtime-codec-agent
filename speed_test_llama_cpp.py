from datetime import datetime
from transformers import AutoTokenizer

from realtime_codec_agent.utils.llamacpp_utils import LlamaForAlternatingCodeChannels

llm = LlamaForAlternatingCodeChannels(
    model_path="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-Q8_0.gguf",
    n_ctx=131072,
    n_gpu_layers=-1,
    verbose=False,
    #flash_attn=True,
    n_threads=24,
    n_threads_batch=24,
    n_batch=2048,
    n_ubatch=512,
)
tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo")

input_ids = tokenizer.encode("<|text_first|><|speaker|>A<|speaker|>B<|end_header|> A: hi how are you?<|audio|>")
num_generate = 4096
single_generate_call = False

start = datetime.now()

llm.init_sampler_for_generate(
    top_k=50,
    top_p=1.0,
    min_p=0.0,
    temp=1.0,
)
output_tokens = [0 for _ in range(num_generate)]
if single_generate_call:
    for i, output in enumerate(llm.generate(input_ids, reset=False)):
        output_tokens[i] = output
        if (i+1) == num_generate:
            break
else:
    for i in range(num_generate):
        output = next(llm.generate(input_ids, reset=False))
        output_tokens[i] = output
        input_ids = [output]
end = datetime.now()
elapsed = end - start
tokens_per_second = len(output_tokens) / elapsed.total_seconds()

output = tokenizer.decode(output_tokens, skip_special_tokens=False)

print(output)
print(f"Time taken: {elapsed.total_seconds():.2f} seconds")
print(f"Tokens generated: {len(output_tokens)}")
print(f"Tok/s: {tokens_per_second:.2f}")