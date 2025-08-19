python -m codec_bpe.audio_to_codes \
    --audio_path data/audio/raw \
    --codes_path data/audio/codes \
    --chunk_size_secs 0.1 \
    --context_secs 2.0 \
    --batch_size 256 \
    --codec_model MagiCodec-50Hz-Base \
    --audio_filter libri-light-medium