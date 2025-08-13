import gradio as gr
import numpy as np
import argparse
import logging
import librosa
from datetime import datetime
from typing import Tuple

from realtime_codec_agent.realtime_agent_v2 import RealtimeAgent, RealtimeAgentResources
from realtime_codec_agent.utils.audio_utils import pad_or_trim

logger = logging.getLogger(__name__)

agent = None

def set_config_and_reset(
    opening_text: str,
    enrollment_audio: Tuple[int, np.ndarray],
    seed: int,
    chunk_size_secs: float,
    temperature: float,
    trans_temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_context_secs: float,
    trim_by_secs: float,
):
    config = agent.config
    config.agent_opening_text = opening_text
    config.agent_voice_enrollment = enrollment_audio
    config.seed = int(seed) if seed else None
    config.chunk_size_secs = float(chunk_size_secs)
    config.temperature = float(temperature)
    config.trans_temperature = float(trans_temperature)
    config.top_k = int(top_k)
    config.top_p = float(top_p)
    config.min_p = float(min_p)
    config.repeat_penalty = float(repeat_penalty)
    config.presence_penalty = float(presence_penalty)
    config.frequency_penalty = float(frequency_penalty)
    config.max_context_secs = float(max_context_secs)
    config.trim_by_secs = float(trim_by_secs)

    if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
        config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

    agent.set_config(config)
    agent.reset()

def run_agent(
    input_audio: Tuple[int, np.ndarray],
    input_channel: int,
    *config_args,
):
    sr, input_audio = input_audio
    if input_audio.ndim == 2:
        input_audio = input_audio[:, input_channel]
    input_audio = input_audio.astype("float32") / 32768.0
    if sr != agent.resources.audio_tokenizer.sampling_rate:
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=agent.resources.audio_tokenizer.sampling_rate)
        sr = agent.resources.audio_tokenizer.sampling_rate

    set_config_and_reset(*config_args)

    report_interval_secs = 2.0
    realtime_factor_sum = 0.0
    report_chunk_count = 0
    last_report_end = 0
    for start in range(0, input_audio.shape[-1], agent.chunk_size_samples):
        end = start + agent.chunk_size_samples
        chunk = input_audio[start:end]
        chunk = pad_or_trim(chunk, agent.chunk_size_samples)

        start_time = datetime.now()
        _ = agent.process_audio(chunk)
        end_time = datetime.now()
        elapsed_secs = (end_time - start_time).total_seconds()
        realtime_factor_sum += agent.config.chunk_size_secs / elapsed_secs
        report_chunk_count += 1

        if report_chunk_count * agent.config.chunk_size_secs >= report_interval_secs or end >= input_audio.shape[-1]:
            realtime_factor = realtime_factor_sum / report_chunk_count
            out_history = agent.get_audio_history()
            transcript = agent.format_transcript()
            sequence = agent.get_sequence_str()
            yield f"{realtime_factor:.2f}x", (sr, out_history[..., last_report_end:end].T), (sr, out_history.T), transcript, sequence
            realtime_factor_sum = 0.0
            report_chunk_count = 0
            last_report_end = end

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the Realtime Codec Agent debug interface.")
    parser.add_argument(
        "--llm_model_path", 
        default="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-2-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-2-F16.gguf", 
        help="Path to the model GGUF file.",
    )

    args = parser.parse_args()
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            llm_model_path=args.llm_model_path,
        ),
    )

    interface = gr.Interface(
        fn=run_agent,
        inputs=[
            gr.Audio(label="Input Audio"),
            gr.Number(0, minimum=0, maximum=1, step=1, label="Input Channel (0=left, 1=right; Ignored if mono)"),
            gr.Textbox("hello how are you?", label="Agent Opening Text"),
            gr.Audio(label="Agent Voice Enrollment"),
            gr.Number(42, minimum=0, step=1, label="Random seed (0 for random)"),
            gr.Slider(0.02, 1.0, value=0.1, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Transcription Temperature (0 for greedy)"),
            gr.Slider(0, 500, value=100, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=0.0, step=0.001, label="Min-p"),
            gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Repeat Penalty"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Frequency Penalty"),
            gr.Number(80.0, minimum=5.0, maximum=80.0, step=5.0, label="Max Context Length (seconds)"),
            gr.Number(20.0, minimum=1.0, maximum=20.0, step=1.0, label="Trim By (seconds)"),
        ], 
        outputs=[
            gr.Textbox(label="Realtime Factor"),
            gr.Audio(label="Audio (streaming)", streaming=True, autoplay=True),
            gr.Audio(label="Audio"),
            gr.TextArea(label="Transcript"),
            gr.TextArea(label="Sequence"),
        ],
        allow_flagging='never'
    )
    interface.launch()
