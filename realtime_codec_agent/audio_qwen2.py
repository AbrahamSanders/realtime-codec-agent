from typing import List, Optional, Tuple, Union
from transformers.models.qwen2 import Qwen2Model, Qwen2ForCausalLM, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from torch import nn
from torch.nn import CrossEntropyLoss
import torch

logger = logging.get_logger(__name__)

class AudioQwen2Config(Qwen2Config):
    def __init__(
        self,
        num_codebooks=2,
        codebook_size=1024,
        codebook_dim=128,
        projector_hidden_act="gelu",
        isolate_codebook_loss=False,
        **kwargs
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.projector_hidden_act = projector_hidden_act
        self.isolate_codebook_loss = isolate_codebook_loss
        super().__init__(**kwargs)

# Adapted from transformers.models.llava.modeling_llava.LlavaMultiModalProjector
class AudioQwen2MultiModalProjector(nn.Module):
    def __init__(self, config: AudioQwen2Config):
        super().__init__()

        self.linear_1 = nn.Linear(config.codebook_dim, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, audio_embeds):
        hidden_states = self.linear_1(audio_embeds)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class AudioQwen2Model(Qwen2Model):
    def __init__(self, config: AudioQwen2Config):
        super(Qwen2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        audio_embed = torch.zeros(config.num_codebooks * config.codebook_size, config.codebook_dim)
        self.register_buffer("audio_embed", audio_embed)
        self.audio_projectors = nn.ModuleList(
            [AudioQwen2MultiModalProjector(config) for _ in range(config.num_codebooks)]
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def embed_audio_tokens(self, audio_input_ids):
        audio_input_ids = audio_input_ids - self.config.vocab_size
        embeds = nn.functional.embedding(audio_input_ids, self.audio_embed, self.padding_idx)
        proj_embeds = torch.empty(embeds.shape[:-1] + (self.config.hidden_size,), dtype=embeds.dtype, device=embeds.device)
        cb_size = self.config.codebook_size
        for i, audio_proj in enumerate(self.audio_projectors):
            codebook_tokens = (audio_input_ids >= i*cb_size) & (audio_input_ids < (i+1)*cb_size)
            proj_embeds[codebook_tokens] = audio_proj(embeds[codebook_tokens]).to(proj_embeds.dtype)
        return proj_embeds
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = torch.empty((batch_size, seq_length, self.config.hidden_size), dtype=self.dtype, device=self.device)
            vocab_tokens = input_ids < self.config.vocab_size
            audio_tokens = input_ids >= self.config.vocab_size
            inputs_embeds[vocab_tokens] = self.embed_tokens(input_ids[vocab_tokens])
            inputs_embeds[audio_tokens] = self.embed_audio_tokens(input_ids[audio_tokens])

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
class AudioQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = AudioQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(config.hidden_size, config.num_codebooks * config.codebook_size, bias=False)

        self.vocab_splits = [
            (0, self.config.vocab_size), 
            (self.config.vocab_size, self.config.vocab_size + config.num_codebooks * config.codebook_size)
        ]

        # Initialize weights and apply final processing
        self.post_init()

    def _compute_loss(self, loss_fct, shift_logits, shift_labels):
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        audio_logits = self.audio_lm_head(hidden_states)
        logits = torch.cat([logits, audio_logits], dim=-1)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            if self.config.isolate_codebook_loss:
                loss = torch.tensor(0., dtype=shift_logits.dtype, device=shift_logits.device)
                # Compute loss over tokens of each vocab split (e.g., text tokens, codebook tokens) individually 
                # (i.e., only logits within each split will contribute to the loss for tokens of that split, 
                #  and token losses will be only be averaged within the split, not across all tokens in the batch)
                num_included_splits = 0
                for start, end in self.vocab_splits:
                    split_tokens = (shift_labels >= start) & (shift_labels < end) & (shift_labels != loss_fct.ignore_index)
                    if split_tokens.any():
                        split_shift_logits = shift_logits[split_tokens][..., start:end]
                        split_shift_labels = shift_labels[split_tokens] - start
                        loss += self._compute_loss(loss_fct, split_shift_logits, split_shift_labels)
                        num_included_splits += 1
                loss /= num_included_splits
            else:
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
                shift_labels = shift_labels.view(-1)
                loss = self._compute_loss(loss_fct, shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )