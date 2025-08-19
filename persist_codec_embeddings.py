from realtime_codec_agent.codec_llama import CodecLlamaForCausalLM
from transformers import AutoModelForCausalLM
from tqdm import trange
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persist codec embeddings in CodecLlamaForCausalLM to the model's main embedding matrix.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained CodecLlamaForCausalLM model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for persisting codec embeddings."
    )
    parser.add_argument(
        "--codec_embed_file",
        type=str,
        default=None,
        help="Path to the codec embedding file. If provided, it will be used to verify the codec embedding matrix."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodecLlamaForCausalLM.from_pretrained(args.model_path).to(device)

    if args.codec_embed_file:
        print("Verifying the codec embedding weight against the provided codec embed file...")
        codec_embed_weight = torch.load(args.codec_embed_file, map_location=device)
        codec_embed_weight = codec_embed_weight.view(-1, codec_embed_weight.shape[-1])
        assert torch.equal(
            model.model.embed_codec_tokens.codec_embed.weight, 
            codec_embed_weight
        ), "Codec embedding weight does not match the provided codec embed file."
    else:
        print("No codec embed file provided, skipping embedding weight verification.")

    print("Persisting codec embeddings to the main embedding matrix...")
    model.model.persist_codec_embeddings(batch_size=args.batch_size)
    print("Codec embeddings persisted successfully. Saving the model...")
    model.save_pretrained(args.model_path)
    print("Model saved.")
    
    # Reload the model and verify the vanilla code embeddings are the same as the projected embeddings
    if device.type == "cuda":
        print("Cleaning up...")
        model.to("cpu")
        torch.cuda.empty_cache()
    print("Reloading the model to verify codec embeddings...")
    model = CodecLlamaForCausalLM.from_pretrained(args.model_path).to(device)
    vanilla_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    num_embeddings = model.config.num_codebooks * model.config.codebook_size
    codec_input_ids = torch.arange(
        model.config.codec_vocab_start, 
        model.config.codec_vocab_start + num_embeddings, 
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        for start in trange(0, num_embeddings, args.batch_size, desc="Verifying codec embeddings"):
            end = start + args.batch_size
            batch_codec_input_ids = codec_input_ids[start:end]
            proj_embeds = model.model.embed_codec_tokens(batch_codec_input_ids)
            vanilla_embeds = vanilla_model.get_input_embeddings()(batch_codec_input_ids)
            assert torch.equal(vanilla_embeds, proj_embeds), "proj_embeds does not match vanilla_embeds"
    
    print("Codec embeddings verified successfully after reload.")

