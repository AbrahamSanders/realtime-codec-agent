import os
import librosa
from tqdm import tqdm

fast = True
totals = {}
num_files = 0
pbar = tqdm(os.walk("data/audio/raw"))
for root, dirs, files in pbar:
    pbar.set_description(f"Processing {root}")
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav") or file.endswith(".opus") or file.endswith(".flac"):
            file_path = os.path.join(root, file)
            if fast:
                duration = librosa.get_duration(path=file_path)
            else:
                audio, sr = librosa.load(file_path)
                duration = audio.shape[-1] / sr
            totals[root] = totals.get(root, 0) + duration
            num_files += 1

total_duration = 0
for folder, duration in totals.items():
    total_duration += duration
    print(f"{folder}: {duration / 3600:.2f} hours")
print(f"Total: {total_duration / 3600:.2f} hours in {num_files} files")