def add_special_audio_tokens(tokenizer, model_vocab_size, num_codebooks, codebook_size):
    if len(tokenizer) > model_vocab_size:
        raise ValueError(f"Tokenizer has {len(tokenizer)} tokens, which is greater than the model's vocab size of {model_vocab_size}.")
    added_tokens = 0
    if len(tokenizer) < model_vocab_size:
        num_pad_tokens = model_vocab_size - len(tokenizer)
        pad_tokens = [f"<p{i:04d}>" for i in range(num_pad_tokens)]
        added_tokens += tokenizer.add_tokens(pad_tokens, special_tokens=True)
    audio_tokens = []
    for i in range(num_codebooks):
        for j in range(codebook_size):
            audio_tokens.append(f"<c{i}t{j:04d}>")
    added_tokens += tokenizer.add_tokens(audio_tokens, special_tokens=True)
    return added_tokens