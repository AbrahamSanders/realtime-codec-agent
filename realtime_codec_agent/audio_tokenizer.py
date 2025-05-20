import torch
import librosa
import numpy as np
import itertools
import math
from typing import Optional, Tuple, Union
from codec_bpe import codes_to_chars, chars_to_codes, UNICODE_OFFSET_LARGE
from xcodec2.modeling_xcodec2 import XCodec2Model

class AudioTokenizer:
    def __init__(
        self, 
        codec_model: Union[str, XCodec2Model] = "HKUSTAudio/xcodec2", 
        num_channels: int = 1, 
        num_codebooks: int = 1,
        codebook_size: int = 65536,
        context_secs: float = 3.0,
        unicode_offset: int = UNICODE_OFFSET_LARGE,
        device: Optional[Union[str, torch.device]] = None, 
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if isinstance(codec_model, str):
            codec_model = XCodec2Model.from_pretrained(codec_model)
        self.codec_model = codec_model.eval().to(self.device)

        self.num_channels = num_channels
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.context_secs = context_secs
        self.unicode_offset = unicode_offset

        self.sampling_rate = self.codec_model.feature_extractor.sampling_rate
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

    def tokenize_audio(self, audio: Union[Tuple[int, np.ndarray], np.ndarray]) -> str:
        if isinstance(audio, np.ndarray):
            orig_sr = self.sampling_rate
        else:
            orig_sr, audio = audio
        audio = audio.astype("float32") / 32768.0
        if self.num_channels == 1 and audio.ndim > 1:
            audio = librosa.to_mono(audio)
        # resample to codec sample rate if needed
        if orig_sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sampling_rate)
            
        # append audio to the context
        self.tokenize_context = np.concatenate((self.tokenize_context, audio.reshape(self.num_channels, -1)), axis=-1)
        # trim the context to the max context size or audio size, whichever is larger
        self.tokenize_context = self.tokenize_context[..., -max(audio.shape[-1], self.context_samples):]
        
        # get audio codes
        input_audio = torch.tensor(self.tokenize_context).to(self.device)
        encoder_outputs = torch.cat(
            [self.codec_model.encode_code(channel.unsqueeze(0), sample_rate=self.sampling_rate) for channel in input_audio], 
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

    def detokenize_audio(self, audio_codes_str: str) -> Tuple[Tuple[int, np.ndarray], str]:
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
        output_audio = torch.cat(
            [self.codec_model.decode_code(ch_codes.unsqueeze(0)) for ch_codes in input_audio_codes], 
            dim=1,
        )

        # discard context audio that comes before the codes we are detokenizing
        audio_secs = self.get_audio_codes_str_secs(audio_codes_str)
        audio_samples = int(audio_secs * self.sampling_rate)
        output_audio = output_audio[..., -audio_samples:]

        # return audio
        output_audio = output_audio[0, 0] if self.num_channels == 1 else output_audio[0]
        return (self.sampling_rate, output_audio.cpu().numpy()), end_hanging
    
    def _drop_hanging_channel_codes(self, audio_str: str) -> Tuple[str, str]:
        div_rem = len(audio_str) % self.num_channels
        if div_rem != 0:
            audio_str = audio_str[:-div_rem]
            end_hanging = audio_str[-div_rem:]
        else:
            end_hanging = ""
        return audio_str, end_hanging
    
    def _compute_framerate(self) -> float:
        audio = torch.zeros(10 * self.sampling_rate).to(self.device)
        audio_codes = self.codec_model.encode_code(audio.unsqueeze(0), sample_rate=self.sampling_rate)
        samples_per_frame = math.ceil(audio.shape[-1] / audio_codes.shape[-1])
        framerate = self.sampling_rate / samples_per_frame
        return framerate
