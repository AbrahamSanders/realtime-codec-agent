import argparse

def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm_model_path", 
        default="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-F16.gguf", 
        help="Path to the model GGUF file.",
    )
    parser.add_argument(
        "--external_llm_repo_id",
        default="ibm-granite/granite-4.0-h-micro-GGUF",
        help="HuggingFace repo ID for the external LLM model to use (if any).",
    )
    parser.add_argument(
        "--external_llm_filename",
        default="*Q4_K_M.gguf",
        help="Filename for the external LLM model to use (if any).",
    )
    parser.add_argument(
        "--external_llm_tokenizer_repo_id",
        default="ibm-granite/granite-4.0-h-micro",
        help="HuggingFace repo ID for the external LLM tokenizer to use (if any).",
    )