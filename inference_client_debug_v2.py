import gradio as gr
import numpy as np
import argparse
import logging
import librosa
from typing import Tuple
import matplotlib.pyplot as plt

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
    force_trans_after_inactivity_secs: float,
    use_whisper: bool,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_context_secs: float,
    trim_by_secs: float,
    target_volume_rms: float,
    force_response_after_inactivity_secs: float,
    use_external_llm: bool,
    external_llm_instructions: str,
    use_external_tts: bool,
    external_tts_prompt_text: str,
    run_profilers: bool,
):
    config = agent.config
    config.agent_opening_text = opening_text
    config.agent_voice_enrollment = enrollment_audio
    config.seed = int(seed) if seed else None
    config.chunk_size_secs = float(chunk_size_secs)
    config.temperature = float(temperature)
    config.trans_temperature = float(trans_temperature)
    config.force_trans_after_inactivity_secs = float(force_trans_after_inactivity_secs)
    config.use_whisper = bool(use_whisper)
    config.top_k = int(top_k)
    config.top_p = float(top_p)
    config.min_p = float(min_p)
    config.repeat_penalty = float(repeat_penalty)
    config.presence_penalty = float(presence_penalty)
    config.frequency_penalty = float(frequency_penalty)
    config.max_context_secs = float(max_context_secs)
    config.trim_by_secs = float(trim_by_secs)
    config.target_volume_rms = float(target_volume_rms)
    config.force_response_after_inactivity_secs = float(force_response_after_inactivity_secs)
    config.use_external_llm = bool(use_external_llm)
    config.external_llm_instructions = external_llm_instructions
    config.use_external_tts = bool(use_external_tts)
    config.external_tts_prompt_text = external_tts_prompt_text
    config.run_profilers = bool(run_profilers)

    if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
        config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

    agent.set_config(config)
    agent.reset()

def run_agent(
    input_audio: Tuple[int, np.ndarray],
    input_channel: int,
    stream_output: bool,
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

    realtime_plot = None
    for start in range(0, input_audio.shape[-1], agent.chunk_size_samples):
        end = start + agent.chunk_size_samples
        chunk = input_audio[start:end]
        chunk = pad_or_trim(chunk, agent.chunk_size_samples)
        _ = agent.process_audio(chunk)

        if end >= input_audio.shape[-1] or end % int(sr * agent.config.profiler_report_interval_secs) == 0:
            if agent.config.run_profilers:
                if realtime_plot is not None:
                    plt.close(realtime_plot)
                realtime_plot = agent.profilers.build_plot()
            if stream_output or end >= input_audio.shape[-1]:
                out_history = agent.get_audio_history()
                transcript = agent.format_transcript()
                sequence = agent.get_sequence_str()
                yield realtime_plot, (sr, out_history.T), transcript, sequence
            elif realtime_plot is not None:
                yield realtime_plot, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the Realtime Codec Agent debug interface.")
    parser.add_argument(
        "--llm_model_path", 
        default="Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test/Llama-3.2-1B-magicodec-no-bpe-multi-131k-stereo-test-F16.gguf", 
        help="Path to the model GGUF file.",
    )
    parser.add_argument(
        "--external_llm_repo_id",
        default="ibm-granite/granite-4.0-h-micro-GGUF",
        help="HuggingFace repo ID for the external LLM model to use (if any).",
    )
    parser.add_argument(
        "--external_llm_filename",
        default="*Q4_K_M.gguf",
        help="Filename for the external LLM model to use (if any).",
    )
    parser.add_argument(
        "--external_llm_tokenizer_repo_id",
        default="ibm-granite/granite-4.0-h-micro",
        help="HuggingFace repo ID for the external LLM tokenizer to use (if any).",
    )

    args = parser.parse_args()
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    agent = RealtimeAgent(
        resources=RealtimeAgentResources(
            llm_model_path=args.llm_model_path,
            external_llm_repo_id=args.external_llm_repo_id,
            external_llm_filename=args.external_llm_filename,
            external_llm_tokenizer_repo_id=args.external_llm_tokenizer_repo_id,
        ),
    )

    config = agent.config
    interface = gr.Interface(
        fn=run_agent,
        inputs=[
            gr.Audio(label="Input Audio"),
            gr.Number(0, minimum=0, maximum=1, step=1, label="Input Channel (0=left, 1=right; Ignored if mono)"),
            gr.Checkbox(True, label="Stream Output"),
            gr.Textbox(config.agent_opening_text, label="Agent Opening Text"),
            gr.Audio(
                (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T) 
                    if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2 
                    else config.agent_voice_enrollment, 
                label="Agent Voice Enrollment",
            ),
            gr.Number(config.seed, minimum=0, step=1, label="Random seed (0 for random)"),
            gr.Slider(0.02, 1.0, value=config.chunk_size_secs, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=config.temperature, step=0.05, label="Temperature"),
            gr.Slider(0.0, 1.0, value=config.trans_temperature, step=0.05, label="Transcription Temperature (0 for greedy)"),
            gr.Slider(0.0, 10.0, value=config.force_trans_after_inactivity_secs, step=0.1, label="Force Transcription After Inactivity (seconds, 0 to disable)"),
            gr.Checkbox(config.use_whisper, label="Use Whisper for Transcription"),
            gr.Slider(0, 500, value=config.top_k, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=config.top_p, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=config.min_p, step=0.001, label="Min-p"),
            gr.Slider(0.0, 2.0, value=config.repeat_penalty, step=0.05, label="Repeat Penalty"),
            gr.Slider(-2.0, 2.0, value=config.presence_penalty, step=0.05, label="Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=config.frequency_penalty, step=0.05, label="Frequency Penalty"),
            gr.Number(config.max_context_secs, minimum=5.0, maximum=80.0, step=5.0, label="Max Context Length (seconds)"),
            gr.Number(config.trim_by_secs, minimum=1.0, maximum=20.0, step=1.0, label="Trim By (seconds)"),
            gr.Slider(0.0, 0.1, value=config.target_volume_rms, step=0.01, label="Volume Normalization (0 to disable)"),
            gr.Slider(0.0, 10.0, value=config.force_response_after_inactivity_secs, step=0.1, label="Force Response After Inactivity (seconds, 0 to disable)"),
            gr.Checkbox(config.use_external_llm, label=f"Use External LLM ({args.external_llm_repo_id})"),
            gr.TextArea(config.external_llm_instructions, label="External LLM Instructions"),
            gr.Checkbox(config.use_external_tts, label="Use External TTS (VoxCPM) for Speech Generation"),
            gr.Textbox(config.external_tts_prompt_text, label="External TTS Voice Enrollment Prompt Text"),
            gr.Checkbox(config.run_profilers, label="Run Profilers"),
        ], 
        outputs=[
            gr.Plot(label="Realtime Factor Plot"),
            gr.Audio(label="Audio"),
            gr.TextArea(label="Transcript"),
            gr.TextArea(label="Sequence"),
        ],
        allow_flagging='never'
    )
    interface.launch()
