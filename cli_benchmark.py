import numpy as np
import argparse
import logging
import librosa
import soundfile as sf
import os
import json
from tqdm import trange

from realtime_codec_agent.realtime_agent_v2 import RealtimeAgent, RealtimeAgentResources, RealtimeAgentConfig
from realtime_codec_agent.utils.audio_utils import pad_or_trim
from realtime_codec_agent.utils.cli_utils import add_common_inference_args

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the CLI-based Realtime Agent benchmarking tool.")
    add_common_inference_args(parser)
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
    parser.add_argument(
        "--use_external_llm",
        action="store_true",
        help="Whether to use an external LLM for response generation.",
    )
    parser.add_argument(
        "--external_llm_instructions_file",
        type=str,
        default=None,
        help="Path to a text file containing instructions for the external LLM.",
    )

    args = parser.parse_args()
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    external_llm_instructions = None
    if args.use_external_llm and args.external_llm_instructions_file is not None:
        with open(args.external_llm_instructions_file, "r", encoding="utf-8") as f:
            external_llm_instructions = f.read()
        print("Loaded external LLM instructions:\n")
        print(external_llm_instructions)

    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            llm_model_path=args.llm_model_path,
            external_llm_repo_id=args.external_llm_repo_id if args.use_external_llm else None,
            external_llm_filename=args.external_llm_filename if args.use_external_llm else None,
            external_llm_tokenizer_repo_id=args.external_llm_tokenizer_repo_id if args.use_external_llm else None,
        ),
        config=RealtimeAgentConfig(
            use_external_llm=args.use_external_llm,
            external_llm_instructions=external_llm_instructions,
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
        f.write("\n\n")
        if agent.config.use_external_llm:
            f.write("---------------------------------------------------------------------------------------\n")
            f.write("-- External LLM Messages:\n")
            f.write("---------------------------------------------------------------------------------------\n")
            f.write(json.dumps(agent.external_llm.messages, indent=4))
            f.write("\n\n")
    audio_history = (audio_history * 32767.0).astype(np.int16)
    sf.write("recordings/output.wav", audio_history.T, sr)

    print("Done!")
