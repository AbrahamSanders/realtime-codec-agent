from transformers import DataCollatorWithPadding

class DataCollatorWithIgnoredPadding(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"].clone()
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return batch
