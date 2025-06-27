python prep_lm_dataset.py \
    --tokenizer=data/audio/output/Llama-3.2-3B-magicodec-no-bpe-131k \
    --codes_path=data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s/mono \
    --transcripts_path=data/transcripts/processed \
    --drop_last \
    --unicode_offset=0xE000 \
    --save_path=data/audio/dataset_multi_no_bpe_magicodec_mono_131k_4096.txt