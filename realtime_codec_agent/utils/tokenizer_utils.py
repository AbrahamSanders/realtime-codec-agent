def add_special_audio_tokens(tokenizer, use_n_codebooks, codebook_size):
    audio_tokens = []
    for i in range(use_n_codebooks):
        for j in range(codebook_size):
            audio_tokens.append(f"<c{i}t{j:04d}>")
    added_tokens = tokenizer.add_tokens(audio_tokens, special_tokens=True)
    return added_tokens