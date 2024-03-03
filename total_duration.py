import os
import librosa

totals = {}
num_files = 0
for root, dirs, files in os.walk("data/audio/raw"):
    for file in files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(root, file)
            folder = os.path.basename(root)
            duration = librosa.get_duration(path=file_path)
            totals[folder] = totals.get(folder, 0) + duration
            num_files += 1

total_duration = 0
for folder, duration in totals.items():
    total_duration += duration
    print(f"{folder}: {duration / 3600:.2f} hours")
print(f"Total: {total_duration / 3600:.2f} hours in {num_files} files")