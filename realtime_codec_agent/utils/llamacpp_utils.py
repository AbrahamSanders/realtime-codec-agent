from llama_cpp import Llama, LogitsProcessorList, StoppingCriteriaList, LlamaGrammar
from typing import Sequence, Optional, Generator
import sys
import ctypes

class LlamaForAlternatingCodeChannels(Llama):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_logprobs(self, ctx_input_ids, input_ids):
        self.reset()
        self.eval(ctx_input_ids)
        self.eval(input_ids)
        logits = self._scores[-len(input_ids)-1:-1]
        logprobs = Llama.logits_to_logprobs(logits)
        logprobs = logprobs[range(len(input_ids)), input_ids]
        return logprobs

    def init_sampler_for_generate(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.set_seed(seed if seed is not None else -1)
        # Reset mirostat sampling
        self._mirostat_mu = ctypes.c_float(2.0 * mirostat_tau)
        self._sampler = self._init_sampler(
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            penalize_nl=penalize_nl,
            logits_processor=logits_processor,
            grammar=grammar,
        )

    def sample(
        self,
        idx: Optional[int] = None,
    ):
        """Sample a token from the model.

        Returns:
            The sampled token.
        """
        assert self.n_tokens > 0

        ridx = idx - self.n_tokens if idx is not None else -1

        assert self.ctx is not None
        assert self._sampler is not None
        token = self._sampler.sample(self._ctx, ridx)
        return token

    def generate(
        self,
        tokens: Sequence[int],
        reset: bool = True,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        # Check for kv cache prefix match
        if reset and self.n_tokens > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix
                if self.verbose:
                    print(
                        f"Llama.generate: {longest_prefix} prefix-match hit, "
                        f"remaining {len(tokens)} prompt tokens to eval",
                        file=sys.stderr,
                    )

        # Reset the model state
        if reset:
            self.reset()

        # # Reset the grammar
        # if grammar is not None:
        #     grammar.reset()

        sample_idx = self.n_tokens + len(tokens) - 1
        tokens = list(tokens)

        # Eval and sample
        while True:
            self.eval(tokens)
            while sample_idx < self.n_tokens:
                token = self.sample(
                    idx=sample_idx,
                )

                sample_idx += 1
                if stopping_criteria is not None and stopping_criteria(
                    self._input_ids[: sample_idx], self._scores[sample_idx - self.n_tokens, :]
                ):
                    return
                tokens_or_none = yield token
                tokens.clear()
                tokens.append(token)
                if tokens_or_none is not None:
                    tokens.extend(tokens_or_none)

                if sample_idx < self.n_tokens and token != self._input_ids[sample_idx]:
                    self.n_tokens = sample_idx
                    self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
                    break

            if self.draft_model is not None:
                self.input_ids[self.n_tokens : self.n_tokens + len(tokens)] = tokens
                draft_tokens = self.draft_model(
                    self.input_ids[: self.n_tokens + len(tokens)]
                )
                tokens.extend(
                    draft_tokens.astype(int)[
                        : self._n_ctx - self.n_tokens - len(tokens)
                    ]
                )