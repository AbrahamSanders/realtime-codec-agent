import argparse
import logging
import librosa
import os
from tqdm import trange

from transformers import AutoTokenizer
from realtime_codec_agent.audio_tokenizer import AudioTokenizer
from realtime_codec_agent.realtime_agent_config import RealtimeAgentConfig
from realtime_codec_agent.realtime_agent_profiler import RealtimeAgentProfilerCollection
from realtime_codec_agent.utils.audio_utils import pad_or_trim

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the CLI-based Audio Tokenization benchmarking tool.")
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

    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(args.llm_model_path))
    audio_tokenizer = AudioTokenizer()

    config = RealtimeAgentConfig(run_profilers=True)
    chunk_size_samples = int(config.chunk_size_secs * audio_tokenizer.sampling_rate)
    profilers = RealtimeAgentProfilerCollection(config)

    input_audio, sr = librosa.load(args.input_audio_path, sr=audio_tokenizer.sampling_rate, mono=False)
    input_audio = input_audio[args.input_channel]

    for start in trange(0, input_audio.shape[-1], chunk_size_samples, desc="Running"):
        end = start + chunk_size_samples
        chunk = input_audio[start:end]
        chunk = pad_or_trim(chunk, chunk_size_samples)
        with profilers.total_profiler:
            with profilers.audio_tokenize_profiler:
                audio_str = audio_tokenizer.tokenize_audio(chunk)
            with profilers.tokenize_profiler:
                input_ids = tokenizer(audio_str, add_special_tokens=False, return_tensors="pt").input_ids
            with profilers.detokenize_profiler:
                out_audio_str = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            with profilers.audio_detokenize_profiler:
                _ = audio_tokenizer.detokenize_audio(out_audio_str)

    fig = profilers.build_plot(ylim=(0.5, 3.0))
    fig.savefig("realtime_factor_profile_scaled1.png")
    fig = profilers.build_plot(ylim=(0.5, 15.0))
    fig.savefig("realtime_factor_profile_scaled2.png")
    fig = profilers.build_plot(ylim=(8.0, 13.0))
    fig.savefig("realtime_factor_profile_scaled3.png")
    fig = profilers.build_plot(ylim=(None, None))
    fig.savefig("realtime_factor_profile_unscaled.png")

