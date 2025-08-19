from realtime_codec_agent.audio_tokenizer import AudioTokenizer
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract codec embeddings from AudioTokenizer")
    parser.add_argument(
        "--codec_model",
        type=str,
        default="MagiCodec-50Hz-Base",
        help="The codec model to use for extracting embeddings."
    )
    args = parser.parse_args()

    audio_tokenizer = AudioTokenizer(args.codec_model)
    codebook = audio_tokenizer.get_codec_embeddings()
    codebook = codebook.unsqueeze(0).to(torch.float32)
    out_filename = f"codec_embed_{args.codec_model}.pt"
    torch.save(codebook, out_filename)
    print(f"Codec embeddings saved to {out_filename}")
    print("Tensor shape:", codebook.shape)
    print("Tensor dtype:", codebook.dtype)