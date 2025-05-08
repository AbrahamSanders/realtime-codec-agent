import librosa
import soundfile as sf
import os
import subprocess
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sph files to mp3")
    parser.add_argument("--sph2pipe", type=str, default="./sph2pipe", help="Path to sph2pipe")
    parser.add_argument("--audio-dir", type=str, default="data/audio/raw", help="Audio directory containing the sph files")
    parser.add_argument("--verify", action="store_true", help="Verify the converted mp3 files")
    parser.add_argument("--delete", action="store_true", help="Delete the original sph files")
    args = parser.parse_args()

    num_sph_files = 0
    num_mp3_files = 0
    num_deleted_files = 0
    for root, dirs, files in os.walk(args.audio_dir):
        files = sorted([f for f in files if f.endswith(".sph") and not f.endswith("-raw.sph")])
        if len(files) == 0:
            continue
        action = "Converting" if not args.verify else "Verifying"
        print(f"{action} in {root}...")
        for file in tqdm(files, desc="Files"):
            file_path = os.path.join(root, file)
            raw_filepath = file_path.replace(".sph", "-raw.sph")
            mp3_filepath = file_path.replace(".sph", ".mp3")
            if not args.verify and os.path.exists(raw_filepath):
                os.remove(raw_filepath)
            if not args.verify and os.path.exists(mp3_filepath):
                os.remove(mp3_filepath)
            try:
                num_sph_files += 1
                if not args.verify:
                    # Run sph2pipe to make the sph file readable
                    subprocess.run([args.sph2pipe, file_path, raw_filepath], check=True)
                    # Load the raw sph file
                    audio, sr = librosa.load(raw_filepath, sr=None, mono=False)
                    # Output mp3
                    sf.write(mp3_filepath, audio.T, sr, format="mp3")
                    # Remove the -raw.sph file
                    os.remove(raw_filepath)
                    num_mp3_files += 1
                elif os.path.exists(mp3_filepath):
                    # Verified that the mp3 file exists
                    num_mp3_files += 1

                if args.delete:
                    os.remove(file_path)
                    num_deleted_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if not args.verify:
        print(f"Attempted to convert {num_sph_files} SPH files:")
        print(f"{num_mp3_files} Succeeded.")
        print(f"{num_sph_files-num_mp3_files} Failed.")
    else:
        print(f"Num. SPH files: {num_sph_files}")
        print(f"Num. corresponding MP3 files: {num_mp3_files}")
        print(f"Num. SPH files without MP3 files: {num_sph_files-num_mp3_files}")
    if args.delete:
        print(f"Deleted {num_deleted_files} SPH files.")
