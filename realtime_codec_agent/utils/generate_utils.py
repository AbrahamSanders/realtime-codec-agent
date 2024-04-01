import torch
from transformers import LogitsProcessor

class ExcludeCodebooksLogitsProcessor(LogitsProcessor):
    def __init__(self, semantic_vocab_size: int):
        self.semantic_vocab_size = semantic_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.semantic_vocab_size:] = -float("inf")
        return scores