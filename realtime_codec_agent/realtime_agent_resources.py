import torch
import os
from typing import Union, Optional
from transformers import AutoTokenizer

from .audio_tokenizer import AudioTokenizer
from .utils.llamacpp_utils import LlamaForAlternatingCodeChannels

class RealtimeAgentResources:
    def __init__(
        self, 
        llm_model_path: str = "Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-F16.gguf", 
        llm_n_ctx: int = 16384,
        codec_model: str = "MagiCodec-50Hz-Base", 
        codec_device: Optional[Union[str, torch.device]] = None,
        whisper_model: Optional[str] = "small.en",
        external_llm_repo_id: Optional[str] = "ibm-granite/granite-4.0-micro-GGUF",
        external_llm_filename: Optional[str] = "*Q4_K_M.gguf",
        external_llm_tokenizer_repo_id: Optional[str] = "ibm-granite/granite-4.0-micro",
        external_llm_n_ctx: int = 0,
    ):
        self.llm_model_dir = os.path.dirname(llm_model_path)
        self.llm = LlamaForAlternatingCodeChannels(
            model_path=llm_model_path,
            n_ctx=llm_n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            flash_attn=True,
        )
        self.aux_llm = LlamaForAlternatingCodeChannels(
            model_path=llm_model_path,
            n_ctx=llm_n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            flash_attn=True,
            logits_all=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_dir)
        self.audio_tokenizer = AudioTokenizer(codec_model=codec_model, device=codec_device)
        self.whisper_model = whisper_model
        if self.whisper_model is not None:
            from pywhispercpp.model import Model
            self.whisper_model = Model(self.whisper_model)
        self.external_llm = None
        if external_llm_repo_id is not None:
            if external_llm_filename is None:
                raise ValueError("external_llm_filename must be provided if external_llm_repo_id is provided.")
            self.external_llm = LlamaForAlternatingCodeChannels.from_pretrained(
                repo_id=external_llm_repo_id,
                filename=external_llm_filename,
                n_ctx=external_llm_n_ctx,
                n_gpu_layers=-1,
                verbose=False,
                flash_attn=True,
            )
            if external_llm_tokenizer_repo_id is None:
                external_llm_tokenizer_repo_id = external_llm_repo_id
            self.external_llm_tokenizer = AutoTokenizer.from_pretrained(external_llm_tokenizer_repo_id)
