import os
import numpy as np
from tqdm import tqdm
from codec_bpe.core.utils import get_codec_info

codes_path = "data/audio/codes/xcodec2/mono"
codec_info = get_codec_info(codes_path)
framerate = codec_info["framerate"]
print(f"Framerate: {framerate} Hz")
totals = {}
num_files = 0
pbar = tqdm(os.walk(codes_path))
for root, _, files in pbar:
    pbar.set_description(f"Processing {root}")
    for file in files:
        if file.endswith(".npy"):
            file_path = os.path.join(root, file)
            codes = np.load(file_path)
            duration = codes.shape[-1] / framerate
            totals[root] = totals.get(root, 0) + duration
            num_files += 1

total_duration = 0
for folder, duration in totals.items():
    total_duration += duration
    print(f"{folder}: {duration / 3600:.2f} hours")
print(f"Total: {total_duration / 3600:.2f} hours in {num_files} codes files")