python -m codec_bpe.audio_to_codes \
    --audio_path data/audio/raw \
    --codes_path data/audio/codes \
    --chunk_size_secs 0.1 \
    --context_secs 2.0 \
    --batch_size 256 \
    --codec_model MagiCodec-50Hz-Base \
    --audio_filter fisher_eng_tr_sp_LDC2004S13 fe_03_p2_LDC2005S13