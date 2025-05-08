from transformers import DataCollatorWithPadding
import torch

class DataCollatorWithIgnoredPadding(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"].clone()
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return batch

class DataCollatorForAlignedDataset(DataCollatorWithIgnoredPadding):
    def __init__(self, *args, model_vocab_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_vocab_size = model_vocab_size
    
    def __call__(self, features):
        batch = super().__call__(features)
        labels = batch["labels"]
        if self.model_vocab_size is not None:
            for i in range(labels.shape[0]):
                # Ignore audio tokens in loss for alignment examples
                if ((labels[i] != self.tokenizer.pad_token_id) & (labels[i] < self.model_vocab_size)).any():
                    labels[i, labels[i] >= self.model_vocab_size] = -100
        return batch
    
class DataCollatorWithIgnoredCodebooks(DataCollatorWithIgnoredPadding):
    def __init__(self, codebook_size, unicode_offset, *args, mask_attn_after_n_codebooks=None, mask_loss_after_n_codebooks=None, **kwargs):
        super().__init__(*args, **kwargs)
        mask_attn_token_ids = []
        mask_loss_token_ids = []
        if mask_attn_after_n_codebooks or mask_loss_after_n_codebooks:
            for i in range(len(self.tokenizer)):
                token = self.tokenizer.convert_ids_to_tokens(i)
                token = token.lstrip("<").rstrip(">")
                mask_attn = bool(mask_attn_after_n_codebooks)
                mask_loss = bool(mask_loss_after_n_codebooks)
                for c in token:
                    if mask_attn and ord(c) < unicode_offset + codebook_size * mask_attn_after_n_codebooks:
                        mask_attn = False
                    if mask_loss and ord(c) < unicode_offset + codebook_size * mask_loss_after_n_codebooks:
                        mask_loss = False
                    if not mask_attn and not mask_loss:
                        break
                if mask_attn:
                    mask_attn_token_ids.append(i)
                if mask_loss:
                    mask_loss_token_ids.append(i)
        self.mask_attn_token_ids = torch.tensor(mask_attn_token_ids)
        self.mask_loss_token_ids = torch.tensor(mask_loss_token_ids)

    def __call__(self, features):
        batch = super().__call__(features)
        labels = batch["labels"]
        if self.mask_attn_token_ids.shape[0] > 0:
            mask_attn_tokens = torch.isin(labels, self.mask_attn_token_ids)
            attention_mask = batch["attention_mask"]
            attention_mask[mask_attn_tokens] = 0
        if self.mask_loss_token_ids.shape[0] > 0:
            mask_loss_tokens = torch.isin(labels, self.mask_loss_token_ids)
            labels[mask_loss_tokens] = -100
        return batch
        