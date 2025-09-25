import torch
import librosa
import numpy as np
import itertools
import math
from typing import Optional, Tuple, Union, Any
from codec_bpe import codes_to_chars, chars_to_codes, UNICODE_OFFSET_LARGE
from codec_bpe.tools.codec_utils import load_magicodec_model

class AudioTokenizer:
    def __init__(
        self, 
        codec_model: Union[str, Any] = "MagiCodec-50Hz-Base", 
        num_channels: int = 1, 
        context_secs: float = 2.0,
        unicode_offset: int = UNICODE_OFFSET_LARGE,
        device: Optional[Union[str, torch.device]] = None, 
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.autocast_bfloat16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        if isinstance(codec_model, str):
            codec_model, _, _ = load_magicodec_model(codec_model, self.device)
        self.codec_model = codec_model.eval().to(self.device)

        self.num_channels = num_channels
        self.num_codebooks = 1
        self.codebook_size = self.codec_model.codebook_size
        self.context_secs = context_secs
        self.unicode_offset = unicode_offset

        self.sampling_rate = self.codec_model.sample_rate
        self.framerate = self._compute_framerate()

        self.context_samples = int(self.context_secs * self.sampling_rate)
        self.context_frames = int(self.context_secs * self.framerate * self.num_channels)

        self.reset_context()

    def reset_context(self):
        self.tokenize_context = np.zeros((self.num_channels, 0), dtype=np.float32)
        self.detokenize_context = ""

    def get_audio_codes_str_secs(self, audio_codes_str: str) -> float:
        secs = len(audio_codes_str) / (self.framerate * self.num_channels)
        return secs

    def chunked_tokenize_audio(self, audio: Union[Tuple[int, np.ndarray], np.ndarray], chunk_size_secs: float) -> str:
        if isinstance(audio, np.ndarray):
            sr = self.sampling_rate
        else:
            sr, audio = audio
        chunk_size_samples = int(chunk_size_secs * sr)
        chunk_codes_strs = []
        for start in range(0, audio.shape[-1], chunk_size_samples):
            end = start + chunk_size_samples
            chunk = audio[..., start:end]
            chunk_codes_str = self.tokenize_audio((sr, chunk))
            chunk_codes_strs.append(chunk_codes_str)
        audio_codes_str = "".join(chunk_codes_strs)
        return audio_codes_str

    @torch.inference_mode()
    def tokenize_audio(self, audio: Union[Tuple[int, np.ndarray], np.ndarray]) -> str:
        audio = self._prep_audio_for_tokenization(audio)
            
        # append audio to the context
        self.tokenize_context = np.concatenate((self.tokenize_context, audio.reshape(self.num_channels, -1)), axis=-1)
        # trim the context to the max context size or audio size, whichever is larger
        self.tokenize_context = self.tokenize_context[..., -max(audio.shape[-1], self.context_samples):]
        
        # get audio codes
        input_audio = torch.tensor(self.tokenize_context).to(self.device)
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = self.autocast_bfloat16,
        ):
            encoder_outputs = torch.cat(
                [self._magicodec_encode(channel.unsqueeze(0)) for channel in input_audio], 
                dim=0,
            )

        # convert to unicode string
        channels_chars = [
            codes_to_chars(
                ch_codes, 
                self.codebook_size, 
                unicode_offset=self.unicode_offset,
            ) for ch_codes in encoder_outputs
        ]
        audio_codes_str = "".join(list(itertools.chain.from_iterable(zip(*channels_chars))))

        # discard context codes that come before the audio we are tokenizing
        audio_secs = audio.shape[-1] / self.sampling_rate
        audio_frames = int(audio_secs * self.framerate * self.num_channels)
        audio_codes_str = audio_codes_str[-audio_frames:]

        return audio_codes_str

    @torch.inference_mode()
    def detokenize_audio(self, audio_codes_str: str, preroll_samples: int = 0) -> Tuple[Tuple[int, np.ndarray], str, int]:
        # make sure len(audio_codes_str) is divisible by num_channels
        audio_codes_str, end_hanging = self._drop_hanging_channel_codes(audio_codes_str)
        
        # append audio codes to the context
        self.detokenize_context += audio_codes_str
        # trim the context to the max context size or audio codes size, whichever is larger
        self.detokenize_context = self.detokenize_context[-max(len(audio_codes_str), self.context_frames):]

        # split the audio codes into channels
        input_audio_codes_str = [self.detokenize_context[i::self.num_channels] for i in range(self.num_channels)]

        # convert unicode string to audio codes
        input_audio_codes = [
            chars_to_codes(
                ch_chars, 
                self.num_codebooks, 
                self.codebook_size,
                return_tensors="pt",
                unicode_offset=self.unicode_offset,
            ) for ch_chars in input_audio_codes_str
        ]
        input_audio_codes = torch.stack(input_audio_codes).to(self.device)
        
        # decode audio codes with codec
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = self.autocast_bfloat16,
        ):
            output_audio = torch.cat(
                [self._magicodec_decode(ch_codes.unsqueeze(0)) for ch_codes in input_audio_codes], 
                dim=1,
            )

        # discard context audio that comes before the codes we are detokenizing
        audio_secs = self.get_audio_codes_str_secs(audio_codes_str)
        audio_samples = int(audio_secs * self.sampling_rate) + preroll_samples
        output_audio = output_audio[..., -audio_samples:]
        preroll_samples = max(0, preroll_samples-audio_samples+output_audio.shape[-1])

        # return audio
        output_audio = output_audio[0, 0] if self.num_channels == 1 else output_audio[0]
        return (self.sampling_rate, output_audio.cpu().numpy()), end_hanging, preroll_samples
    
    @torch.inference_mode()
    def get_codec_embeddings(self) -> torch.Tensor:
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = self.autocast_bfloat16,
        ):
            codebook = self.codec_model.quantizer.codebook_proj(self.codec_model.quantizer.codebook.weight)
        return codebook

    def _drop_hanging_channel_codes(self, audio_str: str) -> Tuple[str, str]:
        div_rem = len(audio_str) % self.num_channels
        if div_rem != 0:
            audio_str = audio_str[:-div_rem]
            end_hanging = audio_str[-div_rem:]
        else:
            end_hanging = ""
        return audio_str, end_hanging
    
    @torch.inference_mode()
    def _encode_silence(self, secs: float) -> torch.Tensor:
        audio = torch.zeros(int(secs * self.sampling_rate)).to(self.device)
        with torch.autocast(
            device_type = "cuda",
            dtype = torch.bfloat16,
            enabled = self.autocast_bfloat16,
        ):
            audio_codes = self._magicodec_encode(audio.unsqueeze(0))
        return audio_codes

    def _compute_framerate(self) -> float:
        test_secs = 10.0
        audio_codes = self._encode_silence(test_secs)
        samples = int(test_secs * self.sampling_rate)
        samples_per_frame = math.ceil(samples / audio_codes.shape[-1])
        framerate = self.sampling_rate / samples_per_frame
        return framerate

    def _magicodec_encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.codec_model.pad_audio(x)
        z_e = self.codec_model.encoder(x)
        _, quantized_indices = self.codec_model.quantizer.inference(z_e)
        quantized_indices = quantized_indices.unsqueeze(1) # add codebook dimension
        return quantized_indices
    
    def _magicodec_decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.squeeze(1) # remove codebook dimension
        codebook = self.codec_model.quantizer.codebook_proj(self.codec_model.quantizer.codebook.weight)
        z_q = torch.nn.functional.embedding(codes, codebook)
        recon = self.codec_model.decoder(z_q).float()
        return recon
    
    def _prep_audio_for_tokenization(self, audio: Union[Tuple[int, np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            orig_sr = self.sampling_rate
        else:
            orig_sr, audio = audio
        if audio.dtype == np.int16:
            audio = audio.astype("float32") / 32768.0
        if self.num_channels == 1 and audio.ndim > 1:
            audio = librosa.to_mono(audio)
        # resample to codec sample rate if needed
        if orig_sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sampling_rate)
        return audio