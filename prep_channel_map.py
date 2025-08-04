import argparse
import os
import librosa
import soundfile as sf
import numpy as np
import json
from codec_bpe.core.utils import get_files
from codec_bpe.tools.audio_encoder import SUPPORTED_EXTENSIONS
from tqdm import tqdm

from realtime_codec_agent.utils.transcript_utils import load_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct a mapping between transcript speaker ids and the audio channel that they are on"
    )
    parser.add_argument("--transcripts_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing channel map files")
    args = parser.parse_args()

    transcript_files = get_files(args.transcripts_path, ".txt")

    for transcript_file in tqdm(transcript_files, desc="Transcript files"):
        audio_file = None
        for ext in SUPPORTED_EXTENSIONS:
            audio_file_ = transcript_file.replace(args.transcripts_path, args.audio_path).replace(".txt", ext)
            if os.path.exists(audio_file_):
                audio_file = audio_file_
                break
        if audio_file is None:
            print(f"Skipping {transcript_file} because no audio file was found.")
            continue
        # open audio file and load transcript
        num_channels = sf.info(audio_file).channels
        if num_channels == 1:
            print(f"Skipping {transcript_file} because audio is mono.")
            continue
        channel_map_file = transcript_file.replace(".txt", "_channel_map.json")
        if not args.overwrite and os.path.exists(channel_map_file):
            print(f"Skipping {transcript_file} because channel map already exists.")
            continue
        audio, sr = librosa.load(audio_file, sr=16000, mono=False)
        transcript_lines, speakers = load_transcript(transcript_file)
        
        # construct the channel map
        channel_map = {}
        for speaker in speakers:
            # concatenate the audio segments for this speaker
            speaker_segments = [line for line in transcript_lines if line[2] == speaker]
            speaker_audio = np.concatenate([
                audio[:, int(start * sr):int(end * sr)] for start, end, _, _ in speaker_segments
            ], axis=-1)
            if speaker_audio.size == 0:
                channel_map[speaker] = {
                    "channel": None,
                    "duration_secs": 0.0,
                }
            else:
                # get average amplitude for each channel during these segments
                speaker_abs_mean_amplitudes = np.mean(np.abs(speaker_audio), axis=-1)
                speaker_channel = np.argmax(speaker_abs_mean_amplitudes)
                channel_map[speaker] = {
                    "channel": speaker_channel.item(),
                    "duration_secs": speaker_audio.shape[-1] / sr,
                }

        # save the channel map to a JSON file alongside the transcript file
        with open(channel_map_file, "w") as f:
            json.dump(channel_map, f, indent=4)
