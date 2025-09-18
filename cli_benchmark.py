import numpy as np
import argparse
import logging
import librosa
import soundfile as sf
import os
from tqdm import trange

from realtime_codec_agent.realtime_agent_v2 import RealtimeAgent, RealtimeAgentResources
from realtime_codec_agent.utils.audio_utils import pad_or_trim

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the CLI-based Realtime Agent benchmarking tool.")
    parser.add_argument(
        "--llm_model_path", 
        default="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-F16.gguf", 
        help="Path to the model GGUF file.",
    )
    parser.add_argument(
        "--input_audio_path", 
        default="data/audio/raw/fisher_eng_tr_sp_LDC2004S13/fisher_eng_tr_sp_d1/audio/000/fe_03_00002.mp3", 
        help="Path to the input audio file.",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=0,
        help="Channel of the input audio to use.",
    )

    args = parser.parse_args()
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            llm_model_path=args.llm_model_path,
        ),
    )

    # load the input audio
    input_audio, sr = librosa.load(args.input_audio_path, sr=agent.resources.audio_tokenizer.sampling_rate, mono=False)
    input_audio = input_audio[args.input_channel]

    # run the agent on the input audio
    for start in trange(0, input_audio.shape[-1], agent.chunk_size_samples, desc="Running"):
        end = start + agent.chunk_size_samples
        chunk = input_audio[start:end]
        chunk = pad_or_trim(chunk, agent.chunk_size_samples)
        _ = agent.process_audio(chunk)

    # save the profiler plots
    fig = agent.profilers.build_plot(ylim=(0.5, 3.0))
    fig.savefig("realtime_factor_profile_scaled1.png")
    fig = agent.profilers.build_plot(ylim=(0.5, 15.0))
    fig.savefig("realtime_factor_profile_scaled2.png")
    fig = agent.profilers.build_plot(ylim=(8.0, 13.0))
    fig.savefig("realtime_factor_profile_scaled3.png")
    fig = agent.profilers.build_plot(ylim=(None, None))
    fig.savefig("realtime_factor_profile_unscaled.png")

    # save the audio and transcript
    audio_history = agent.get_audio_history()
    transcript = agent.format_transcript()
    sequence = agent.get_sequence_str()
    os.makedirs("recordings", exist_ok=True)
    with open("recordings/output.txt", "w", encoding="utf-8") as f:
        f.write("---------------------------------------------------------------------------------------\n")
        f.write("-- Transcript:\n")
        f.write("---------------------------------------------------------------------------------------\n")
        f.write(transcript)
        f.write("\n\n")
        f.write("---------------------------------------------------------------------------------------\n")
        f.write("-- Sequence:\n")
        f.write("---------------------------------------------------------------------------------------\n")
        f.write(sequence)
        f.write("\n")
    audio_history = (audio_history * 32767.0).astype(np.int16)
    sf.write("recordings/output.wav", audio_history.T, sr)

    print("Done!")
