import torch
from transformers import LogitsProcessor

class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of [`Bark`]'s fine submodel.

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    """

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[-1]

        # even -> first codebook, odd -> second codebook
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        if is_first_codebook:
            scores[:, : self.semantic_vocab_size] = -float("inf")
            scores[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")
            scores[:, self.semantic_vocab_size + 2*self.codebook_size :] = -float("inf")

        return scores