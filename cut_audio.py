import librosa
import soundfile as sf
import argparse
from os import path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Truncate audio file to a given length")
    parser.add_argument("--filename", type=str, default="data/audio/raw/CallFriend_eng_n/6899.mp3")
    parser.add_argument("--keep_secs", type=int, default=10)
    args = parser.parse_args()

    audio, sr = librosa.load(args.filename, mono=False)

    audio = audio[:, :args.keep_secs*sr]

    out_filename = path.splitext(path.basename(args.filename))[0]
    sf.write(f"{out_filename}_cut.wav", audio.T, sr)