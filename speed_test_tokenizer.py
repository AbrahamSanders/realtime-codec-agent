import librosa
import time
import numpy as np
import soundfile as sf
from tqdm import trange
from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim

audio_tokenizer = AudioTokenizer()

audio_file = "experimental/audio (4) (1).wav"
audio, _ = librosa.load(audio_file, sr=audio_tokenizer.sampling_rate, mono=True)

chunk_size_secs = 0.1
if int(chunk_size_secs*100) % 2 != 0:
    raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
chunk_size_samples = int(chunk_size_secs * audio_tokenizer.sampling_rate)
out_audio = np.zeros((0,), dtype=np.float32)

crossfade_ramps = create_crossfade_ramps(audio_tokenizer.sampling_rate, fade_secs=0.02) 

start_time = time.time()
for start in trange(0, len(audio), chunk_size_samples, desc="Chunks"):
    end = start + chunk_size_samples
    audio_chunk = audio[start:end]
    audio_chunk_str = audio_tokenizer.tokenize_audio(audio_chunk)
    (_, out_chunk), _, preroll_samples = audio_tokenizer.detokenize_audio(audio_chunk_str, preroll_samples=crossfade_ramps[0])
    out_chunk = pad_or_trim(out_chunk, audio_chunk.shape[-1] + preroll_samples)
    out_audio = smooth_join(out_audio, out_chunk, *crossfade_ramps)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")

output_file = "output.wav"
sf.write(output_file, out_audio, audio_tokenizer.sampling_rate)