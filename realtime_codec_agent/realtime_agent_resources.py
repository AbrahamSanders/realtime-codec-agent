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
