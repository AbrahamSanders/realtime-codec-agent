from typing import Tuple
import numpy as np

def smooth_join(chunk1: np.ndarray, chunk2: np.ndarray, L: int, fade_in: np.ndarray, fade_out: np.ndarray) -> np.ndarray:
    if chunk1.shape[-1] == 0:
        return chunk2
    if L == 0:
        return np.concatenate((chunk1, chunk2), axis=-1)

    # split tails/heads
    head1, tail1 = chunk1[..., :-L], chunk1[..., -L:]
    head2, tail2 = chunk2[..., :L], chunk2[..., L:]

    # apply ramps
    cross = tail1 * fade_out + head2 * fade_in

    return np.concatenate((head1, cross, tail2), axis=-1)

def create_crossfade_ramps(sr: int, fade_secs: float) -> Tuple[int, np.ndarray, np.ndarray]:
    L = int(sr * fade_secs)
    fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, L, endpoint=False))
    fade_out = fade_in[::-1]
    return L, fade_in, fade_out

def pad_or_trim(chunk: np.ndarray, target_length: int, pad_side: str = "right") -> np.ndarray:
    if chunk.ndim > 1:
        raise ValueError("Input chunk must be a 1D array.")
    if chunk.shape[-1] < target_length:
        # Pad with zeros at the end
        pad_width = target_length-chunk.shape[-1]
        pad_width = (0, pad_width) if pad_side == "right" else (pad_width, 0)
        return np.pad(chunk, pad_width, mode="constant")
    elif chunk.shape[-1] > target_length:
        # Trim to target length
        return chunk[..., :target_length]
    else:
        return chunk
    
def normalize_audio_rms(audio, target_rms=0.05, silence_rms_threshold=0.003):
    # Compute current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < silence_rms_threshold:
        # Silent audio, nothing to normalize
        return audio
    # Scale so RMS becomes target_rms
    return audio * (target_rms / rms)