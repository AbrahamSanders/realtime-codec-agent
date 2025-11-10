from typing import Tuple
import gradio as gr
import numpy as np
import argparse
import logging

from realtime_codec_agent import RealtimeAgent, RealtimeAgentResources, add_common_inference_args
from realtime_codec_agent.external_llm_client import ExternalLLMClient

logger = logging.getLogger(__name__)

agent_1 = None
agent_2 = None

def set_config_and_reset(
    opening_text_1: str,
    enrollment_audio_1: Tuple[int, np.ndarray],
    opening_text_2: str,
    enrollment_audio_2: Tuple[int, np.ndarray],
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
    use_external_llm_1: bool,
    external_llm_instructions_1: str,
    use_external_llm_2: bool,
    external_llm_instructions_2: str,
    use_external_tts: bool,
    external_tts_prompt_text: str,
    constrain_allow_noise: bool,
    constrain_allow_breathing: bool,
    constrain_allow_laughter: bool,
    run_profilers: bool,
):
    for i, agent in enumerate([agent_1, agent_2]):
        config = agent.config
        config.agent_opening_text = opening_text_1 if i == 0 else opening_text_2
        config.agent_voice_enrollment = enrollment_audio_1 if i == 0 else enrollment_audio_2
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
        config.use_external_llm = bool(use_external_llm_1 if i == 0 else use_external_llm_2)
        config.external_llm_instructions = external_llm_instructions_1 if i == 0 else external_llm_instructions_2
        config.use_external_tts = bool(use_external_tts)
        config.external_tts_prompt_text = external_tts_prompt_text
        config.constrain_allow_noise = bool(constrain_allow_noise)
        config.constrain_allow_breathing = bool(constrain_allow_breathing)
        config.constrain_allow_laughter = bool(constrain_allow_laughter)
        config.run_profilers = bool(run_profilers)

        if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
            config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

        agent.set_config(config)
        agent.reset()

def run_agent(
    duration_secs: float,
    stream_output: bool,
    *config_args,
):
    set_config_and_reset(*config_args)
    last_stream_secs = 0.0
    out_chunk_1 = out_chunk_2 = (np.zeros(agent_1.chunk_size_samples, dtype=np.float32), None)
    while agent_1.total_secs < float(duration_secs):
        out_chunk_1_ = agent_1.process_audio(*out_chunk_2)
        out_chunk_2 = agent_2.process_audio(*out_chunk_1)
        out_chunk_1 = out_chunk_1_
        if agent_1.total_secs >= float(duration_secs) or (stream_output and agent_1.total_secs - last_stream_secs >= 2.0):
            output = []
            for agent in [agent_1, agent_2]:
                out_history = agent.get_audio_history()
                transcript = agent.format_transcript()
                sequence = agent.get_sequence_str()
                external_llm_messages = agent.get_external_llm_messages()
                output.extend([(agent.resources.audio_tokenizer.sampling_rate, out_history.T), transcript, sequence, external_llm_messages])
            yield output
            last_stream_secs = agent_1.total_secs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the Realtime Codec Agent Self Play debug interface.")
    add_common_inference_args(parser)
    
    args = parser.parse_args()
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    agent_1 = RealtimeAgent(
        resources=RealtimeAgentResources(llm_model_path=args.llm_model_path),
        self_play_mode=True,
    )
    agent_2 = RealtimeAgent(
        resources=RealtimeAgentResources(llm_model_path=args.llm_model_path),
        self_play_mode=True,
    )

    config = agent_1.config
    external_llm_models = ExternalLLMClient.get_models(config.external_llm_api_key, config.external_llm_base_url)
    external_llm_model = external_llm_models[0].split("/")[-1] if len(external_llm_models) > 0 else None
    interface = gr.Interface(
        fn=run_agent,
        inputs=[
            gr.Slider(10, 600, value=60, step=10, label="Self Play Duration (seconds)"),
            gr.Checkbox(True, label="Stream Output"),
            gr.Textbox(config.agent_opening_text, label="Agent 1 Opening Text"),
            gr.Audio(
                (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T) 
                    if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2 
                    else config.agent_voice_enrollment, 
                label="Agent 1 Voice Enrollment",
            ),
            gr.Textbox(config.agent_opening_text, label="Agent 2 Opening Text"),
            gr.Audio(
                (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T) 
                    if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2 
                    else config.agent_voice_enrollment, 
                label="Agent 2 Voice Enrollment",
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
            gr.Checkbox(config.use_external_llm, label=f"Agent 1 Use External LLM ({external_llm_model})", interactive=bool(external_llm_model)),
            gr.TextArea(config.external_llm_instructions, label="Agent 1 External LLM Instructions"),
            gr.Checkbox(config.use_external_llm, label=f"Agent 2 Use External LLM ({external_llm_model})", interactive=bool(external_llm_model)),
            gr.TextArea(config.external_llm_instructions, label="Agent 2 External LLM Instructions"),
            gr.Checkbox(config.use_external_tts, label="Use External TTS (VoxCPM) for Speech Generation"),
            gr.Textbox(config.external_tts_prompt_text, label="External TTS Voice Enrollment Prompt Text"),
            gr.Checkbox(config.constrain_allow_noise, label="Constraint: Allow Noise"),
            gr.Checkbox(config.constrain_allow_breathing, label="Constraint: Allow Breathing"),
            gr.Checkbox(config.constrain_allow_laughter, label="Constraint: Allow Laughter"),
            gr.Checkbox(config.run_profilers, label="Run Profilers"),
        ], 
        outputs=[
            gr.Audio(label="Audio 1"),
            gr.TextArea(label="Transcript 1"),
            gr.TextArea(label="Sequence 1"),
            gr.JSON(label="External LLM Messages 1"),
            gr.Audio(label="Audio 2"),
            gr.TextArea(label="Transcript 2"),
            gr.TextArea(label="Sequence 2"),
            gr.JSON(label="External LLM Messages 2"),
        ],
        allow_flagging='never'
    )
    interface.launch()
