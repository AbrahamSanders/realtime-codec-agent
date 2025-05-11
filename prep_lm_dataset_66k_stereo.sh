python prep_lm_dataset.py \
    --tokenizer=data/audio/output/Llama-3.2-3B-xcodec2-no-bpe-66k \
    --codes_path=data/audio/codes/xcodec2/stereo \
    --transcripts_path=data/transcripts/processed \
    --sequence_length=8192 \
    --overlap_length=2048 \
    --drop_last \
    --unicode_offset=0xE000 \
    --save_path=data/audio/dataset_multi_no_bpe_xcodec2_stereo_66k_8192.txt