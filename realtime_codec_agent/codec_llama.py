import torch
from typing import Optional
from torch import nn
from transformers.models.llama import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.activations import ACT2FN
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack

class CodecLlamaConfig(LlamaConfig):
    def __init__(
        self,
        num_codebooks=1,
        codebook_size=131072,
        codebook_dim=16,
        projector_hidden_act="gelu",
        codec_vocab_start=0,
        **kwargs
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.projector_hidden_act = projector_hidden_act
        self.codec_vocab_start = codec_vocab_start
        super().__init__(**kwargs)

# Adapted from transformers.models.llava.modeling_llava.LlavaMultiModalProjector
class CodecLlamaMultiModalProjector(nn.Module):
    def __init__(self, config: CodecLlamaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.codebook_dim, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, codec_token_embeds):
        hidden_states = self.linear_1(codec_token_embeds)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class CodecLlamaCodecEmbedding(nn.Module):
    def __init__(self, config: CodecLlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.codec_embed = nn.Embedding(
            num_embeddings=config.num_codebooks * config.codebook_size,
            embedding_dim=config.codebook_dim,
            padding_idx=self.padding_idx,
            _freeze=True,  # Freeze the codec embeddings - it is the projector weights that will be trained!!!
        )
        self.codebook_projectors = nn.ModuleList(
            [CodecLlamaMultiModalProjector(config) for _ in range(config.num_codebooks)]
        )

    def forward(self, codec_input_ids: torch.Tensor) -> torch.Tensor:
        codec_input_ids = codec_input_ids - self.config.codec_vocab_start
        embeds = self.codec_embed(codec_input_ids)
        proj_embeds = torch.empty(codec_input_ids.shape + (self.config.hidden_size,), dtype=embeds.dtype, device=embeds.device)
        cb_size = self.config.codebook_size
        for i, codebook_proj in enumerate(self.codebook_projectors):
            codebook_tokens = (codec_input_ids >= i*cb_size) & (codec_input_ids < (i+1)*cb_size)
            proj_embeds[codebook_tokens] = codebook_proj(embeds[codebook_tokens]).to(proj_embeds.dtype)
        return proj_embeds

# Adapted from https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/llama/modeling_llama.py
@auto_docstring
class CodecLlamaModel(LlamaModel):
    def __init__(self, config: CodecLlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_codec_tokens = CodecLlamaCodecEmbedding(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = torch.empty(input_ids.shape + (self.config.hidden_size,), dtype=self.dtype, device=self.device)
            vocab_tokens = input_ids < self.config.codec_vocab_start
            codec_tokens = input_ids >= self.config.codec_vocab_start
            inputs_embeds[vocab_tokens] = self.embed_tokens(input_ids[vocab_tokens])
            inputs_embeds[codec_tokens] = self.embed_codec_tokens(input_ids[codec_tokens])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
@auto_docstring
class CodecLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: CodecLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = CodecLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_codec_embeddings(self, codec_embed_weight: torch.Tensor):
        assert (
            codec_embed_weight.shape == self.model.embed_codec_tokens.codec_embed.weight.shape and
            codec_embed_weight.dtype == self.model.embed_codec_tokens.codec_embed.weight.dtype
        ), (f"codec_embed_weight must be a {self.model.embed_codec_tokens.codec_embed.weight.dtype} tensor "
            f"of shape {self.model.embed_codec_tokens.codec_embed.weight.shape}")
        
        codec_embed_weight = codec_embed_weight.clone(memory_format=torch.contiguous_format).to(
            self.model.embed_codec_tokens.codec_embed.weight.device
        )
        self.model.embed_codec_tokens.codec_embed.weight.data = codec_embed_weight

    def persist_codec_embeddings(self, batch_size: int = 1024, show_progress: bool = True):
        # first we have to untie the embeddings from the LM head if they are tied, otherwise
        # we end up lobotomizing the region of the LM head that corresponds to the codec tokens!
        if getattr(self.config.get_text_config(decoder=True), "tie_word_embeddings"):
            setattr(self.config.get_text_config(decoder=True), "tie_word_embeddings", False)
            self._tied_weights_keys = []
            self.lm_head.weight = torch.nn.Parameter(self.lm_head.weight.clone())

        # now, project each codebook vector and save it in self.embed_tokens
        num_embeddings = self.config.num_codebooks * self.config.codebook_size
        codec_input_ids = torch.arange(
            self.config.codec_vocab_start, 
            self.config.codec_vocab_start + num_embeddings, 
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            if show_progress:
                from tqdm import trange
                range_iter = trange(0, num_embeddings, batch_size, desc="Persisting codec embeddings")
            else:
                range_iter = range(0, num_embeddings, batch_size)
            for start in range_iter:
                end = start + batch_size
                batch_codec_input_ids = codec_input_ids[start:end]
                proj_embeds = self.model.embed_codec_tokens(batch_codec_input_ids)
                self.model.embed_tokens.weight.data[batch_codec_input_ids] = proj_embeds
                # sanity check
                assert torch.equal(self.model.embed_tokens(batch_codec_input_ids), proj_embeds), "proj_embeds does not match embed_tokens"
