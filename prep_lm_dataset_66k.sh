python prep_lm_dataset.py \
    --tokenizer=data/audio/output/Llama-3.2-3B-xcodec2-no-bpe-66k \
    --codes_path=data/audio/codes/xcodec2/mono \
    --transcripts_path=data/transcripts/processed \
    --drop_last \
    --unicode_offset=0xE000 \
    --save_path=data/audio/dataset_multi_no_bpe_xcodec2_mono_66k_4096.txt