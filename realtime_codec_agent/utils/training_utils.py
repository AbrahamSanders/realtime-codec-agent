from transformers import DataCollatorWithPadding

class DataCollatorForAlignedDataset(DataCollatorWithPadding):
    def __init__(self, *args, model_vocab_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_vocab_size = model_vocab_size
    
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"].clone()
        labels = batch["labels"]
        if self.model_vocab_size is not None:
            for i in range(labels.shape[0]):
                # Ignore audio tokens in loss for alignment examples
                if ((labels[i] != self.tokenizer.pad_token_id) & (labels[i] < self.model_vocab_size)).any():
                    labels[i, labels[i] >= self.model_vocab_size] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return batch