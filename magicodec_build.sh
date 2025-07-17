# magicodec
git clone https://github.com/Ereboas/MagiCodec.git

# flash-attn build
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 92dd570
git submodule sync --recursive
git submodule update --init --recursive
python setup.py install

# ops build
cd csrc
cd rotary && python setup.py install && cd ..
cd layer_norm && python setup.py install && cd ..
cd fused_dense_lib && python setup.py install && cd ..