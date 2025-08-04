python prep_lm_dataset.py \
    --codes_path=data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s/stereo \
    --transcripts_path=data/transcripts/processed \
    --unicode_offset=0xE000 \
    --save_path=data/audio/dataset_multi_no_bpe_magicodec_stereo_131k_80s.txt \
    --codes_filter_exclude CallHome_eng/4156 CallHome_eng/4183 CallHome_eng/4484 CallHome_eng/4852 CallFriend_eng_n/5220