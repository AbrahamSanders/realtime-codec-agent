# read in the checkpoint directory from the first command line argument
CHECKPOINT_DIR=$1
TEST_MODEL_DIR="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test"

# if Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test directory exists, remove it
if [ -d $TEST_MODEL_DIR ]; then
  rm -rf $TEST_MODEL_DIR
fi

# create it again
mkdir $TEST_MODEL_DIR

# copy the files from the checkpoint directory
cp $CHECKPOINT_DIR/*.json $TEST_MODEL_DIR
cp $CHECKPOINT_DIR/*.safetensors $TEST_MODEL_DIR

# run the persist script
python persist_codec_embeddings.py \
    --model_path $TEST_MODEL_DIR \
    --codec_embed_file codec_embed_MagiCodec-50Hz-Base.pt \
    --save_vanilla

# delete the original folder and rename the vanilla folder
rm -rf $TEST_MODEL_DIR
mv "${TEST_MODEL_DIR}-vanilla" $TEST_MODEL_DIR

# convert to gguf format
python convert_hf_to_gguf.py $TEST_MODEL_DIR
python convert_hf_to_gguf.py $TEST_MODEL_DIR --outtype q8_0
python convert_hf_to_gguf.py $TEST_MODEL_DIR --outtype f32
../llama.cpp/build/bin/llama-quantize "${TEST_MODEL_DIR}/${TEST_MODEL_DIR}-F32.gguf" "${TEST_MODEL_DIR}/${TEST_MODEL_DIR}-Q4_K_M.gguf" Q4_K_M