from typing import List
from .audio_tokenizer import AudioTokenizer
from transformers import AutoConfig
import torch

class ExternalTTSDuplexAligner:
    def __init__(self, audio_tokenizer: AudioTokenizer, duplex_model_dir: str):
        self.codec_embeddings = audio_tokenizer.get_codec_embeddings()
        duplex_model_config = AutoConfig.from_pretrained(duplex_model_dir)
        self.codec_vocab_start = duplex_model_config.codec_vocab_start

        # get silence embedding
        silence_codes = audio_tokenizer._encode_silence(10.0)[0, 0]
        silence_embeddings = torch.nn.functional.embedding(silence_codes, self.codec_embeddings)
        self.silence_embedding = silence_embeddings.mean(0)

    def interrupt_score(self, tts_token_ids: List[int], duplex_token_ids: List[int]) -> float:
        tts_duplex_codes = torch.tensor([tts_token_ids, duplex_token_ids]).to(self.codec_embeddings.device)
        tts_duplex_codes -= self.codec_vocab_start
        tts_duplex_embs = torch.nn.functional.embedding(tts_duplex_codes, self.codec_embeddings)
        dist_from_silence = torch.linalg.vector_norm(tts_duplex_embs-self.silence_embedding, dim=-1)
        dist_from_silence = dist_from_silence.mean(dim=-1).tolist()
        tts_dist, duplex_dist = dist_from_silence
        # score is the ratio of the distances from silence.
        # interpret as: the tts prediction is {score} times further from silence than the duplex prediction.
        score = tts_dist / (duplex_dist + 1e-5)
        return score