import torch
from transformers import LogitsProcessor
from transformers.generation.streamers import BaseStreamer
from queue import Queue

class ExcludeCodebooksLogitsProcessor(LogitsProcessor):
    def __init__(self, semantic_vocab_size: int):
        self.semantic_vocab_size = semantic_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.semantic_vocab_size:] = -float("inf")
        return scores
    
class ExcludeTextLogitsProcessor(LogitsProcessor):
    def __init__(self, semantic_vocab_size: int):
        self.semantic_vocab_size = semantic_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, :self.semantic_vocab_size] = -float("inf")
        return scores
    
class CodecBPEIteratorStreamer(BaseStreamer):
    def __init__(self, tokenizer, num_codebooks, codec_framerate, skip_prompt = False, flush_secs=3, timeout=None, **decode_kwargs):
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.codec_framerate = codec_framerate
        self.skip_prompt = skip_prompt
        self.flush_secs = flush_secs
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.unicode_cache = []
        self.cache_secs = 0
        self.next_tokens_are_prompt = True

        # iterator variables
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("CodecBPEIteratorStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache update the cumulative duration
        tok_text = self.tokenizer.decode(value, **self.decode_kwargs)
        self.unicode_cache.append(tok_text)
        num_units = len(tok_text) / self.num_codebooks
        num_seconds = num_units / self.codec_framerate
        self.cache_secs += num_seconds

        # After the specified number of seconds, we flush the cache.
        if self.cache_secs >= self.flush_secs:
            unicode_text = "".join(self.unicode_cache)
            self.unicode_cache = []
            self.cache_secs = 0
            self.on_finalized_text(unicode_text)

    def end(self):
        # Flush the cache, if it exists
        if len(self.unicode_cache) > 0:
            unicode_text = "".join(self.unicode_cache)
            self.unicode_cache = []
            self.cache_secs = 0
        else:
            unicode_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(unicode_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if text:
            self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value