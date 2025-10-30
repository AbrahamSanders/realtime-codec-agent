from fastrtc import StreamHandler, Stream, AdditionalOutputs
import logging
import gradio as gr
import numpy as np
import soundfile as sf
import argparse
import os
import json

from realtime_codec_agent import RealtimeAgentMultiprocessing, add_common_inference_args
from realtime_codec_agent.external_llm_client import ExternalLLMClient

class AgentHandler(StreamHandler):
    def __init__(self, agent_1: RealtimeAgentMultiprocessing, agent_2: RealtimeAgentMultiprocessing):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.started = False
        self.initial_chunk = True
        self.last_realtime_factor = None

        agent_info_1 = self.agent_1.get_info()
        self.chunk_size_samples = agent_info_1.chunk_size_samples
        super().__init__(
            input_sample_rate=agent_info_1.sampling_rate,
            output_sample_rate=agent_info_1.sampling_rate,
        )

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        pass

    def emit(self) -> None:
        if not self.started:
            return None
        #get the next output
        if self.initial_chunk:
            out_chunk_1 = out_chunk_2 = np.zeros(self.chunk_size_samples, dtype=np.float32)
            realtime_factor_1 = realtime_factor_2 = None
            self.initial_chunk = False
        else:
            out_chunk_1, realtime_factor_1 = self.agent_1.next_output(block=True)
            out_chunk_2, realtime_factor_2 = self.agent_2.next_output(block=True)
        out_chunk = np.vstack((out_chunk_1, out_chunk_2)).mean(axis=0, keepdims=True)
        out_chunk = (out_chunk * 32767.0).astype(np.int16)
        print(f'Data from model:{out_chunk.shape, out_chunk.dtype, out_chunk.min(), out_chunk.max()}')

        #queue up the next input
        self.agent_1.queue_input(out_chunk_2)
        self.agent_2.queue_input(out_chunk_1)

        # Report if realtime factor has changed
        realtime_factor = min(realtime_factor_1 or 0.0, realtime_factor_2 or 0.0)
        if realtime_factor != self.last_realtime_factor:
            self.last_realtime_factor = realtime_factor
            return (self.output_sample_rate, out_chunk), AdditionalOutputs(f"{realtime_factor:.2f}x")
        
        return (self.output_sample_rate, out_chunk)

    def copy(self):
        return AgentHandler(self.agent_1, self.agent_2)

    def shutdown(self):
        if not self.started:
            print(">>> Not Started <<<")
            return
        for i, agent in enumerate([self.agent_1, self.agent_2]):
            agent_info = agent.get_info()
            os.makedirs("recordings", exist_ok=True)
            with open(f"recordings/output_self_play_agent_{i+1}.txt", "w", encoding="utf-8") as f:
                f.write("---------------------------------------------------------------------------------------\n")
                f.write("-- Transcript:\n")
                f.write("---------------------------------------------------------------------------------------\n")
                f.write(agent_info.transcript)
                f.write("\n\n")
                f.write("---------------------------------------------------------------------------------------\n")
                f.write("-- Sequence:\n")
                f.write("---------------------------------------------------------------------------------------\n")
                f.write(agent_info.sequence)
                f.write("\n\n")
                if agent_info.config.use_external_llm:
                    f.write("---------------------------------------------------------------------------------------\n")
                    f.write("-- External LLM Messages:\n")
                    f.write("---------------------------------------------------------------------------------------\n")
                    f.write(json.dumps(agent_info.external_llm_messages, indent=4))
                    f.write("\n\n")
            audio_history = (agent_info.audio_history * 32767.0).astype(np.int16)
            sf.write(f"recordings/output_self_play_agent_{i+1}.wav", audio_history.T, self.output_sample_rate)
        
        self.started = False
        print(">>> Stopped <<<")

    def start_up(self) -> None:
        self.set_config_and_reset()
        agent_info_1 = self.agent_1.get_info()
        self.chunk_size_samples = agent_info_1.chunk_size_samples
        self.initial_chunk = True
        self.last_realtime_factor = None
        self.started = True
        print(">>> Started <<<")

    def set_config_and_reset(self):
        if not self.phone_mode:
            self.wait_for_args_sync()
            for i, agent in enumerate([self.agent_1, self.agent_2]):
                agent_info = agent.get_info()
                config = agent_info.config
                config.agent_opening_text = self.latest_args[1 if i == 0 else 3]
                config.agent_voice_enrollment = self.latest_args[2 if i == 0 else 4]
                config.seed = int(self.latest_args[5]) if self.latest_args[5] else None
                config.chunk_size_secs = float(self.latest_args[6])
                config.temperature = float(self.latest_args[7])
                config.trans_temperature = float(self.latest_args[8])
                config.force_trans_after_inactivity_secs = float(self.latest_args[9])
                config.use_whisper = bool(self.latest_args[10])
                config.top_k = int(self.latest_args[11])
                config.top_p = float(self.latest_args[12])
                config.min_p = float(self.latest_args[13])
                config.repeat_penalty = float(self.latest_args[14])
                config.presence_penalty = float(self.latest_args[15])
                config.frequency_penalty = float(self.latest_args[16])
                config.max_context_secs = float(self.latest_args[17])
                config.trim_by_secs = float(self.latest_args[18])
                config.target_volume_rms = float(self.latest_args[19])
                config.force_response_after_inactivity_secs = float(self.latest_args[20])
                config.use_external_llm = bool(self.latest_args[21 if i == 0 else 23])
                config.external_llm_instructions = self.latest_args[22 if i == 0 else 24]
                config.use_external_tts = bool(self.latest_args[25])
                config.external_tts_prompt_text = self.latest_args[26]
                config.constrain_allow_noise = bool(self.latest_args[27])
                config.constrain_allow_breathing = bool(self.latest_args[28])
                config.constrain_allow_laughter = bool(self.latest_args[29])
                config.run_profilers = bool(self.latest_args[30])

                if config.agent_voice_enrollment is not None and config.agent_voice_enrollment[1].ndim == 2:
                    config.agent_voice_enrollment = (config.agent_voice_enrollment[0], config.agent_voice_enrollment[1].T)

                agent.set_config_and_reset(config)
        else:
            self.agent_1.reset()
            self.agent_2.reset()

def display_handler(component1, realtime_factor: str):
    return realtime_factor

def main(args):
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    agent_1 = RealtimeAgentMultiprocessing(
        llm_model_path=args.llm_model_path,
        wait_until_running=False,
    )
    agent_2 = RealtimeAgentMultiprocessing(
        llm_model_path=args.llm_model_path,
        wait_until_running=False,
    )
    agent_1.wait_until_running()
    agent_2.wait_until_running()

    agent_info_1 = agent_1.get_info()
    config = agent_info_1.config
    external_llm_models = ExternalLLMClient.get_models(config.external_llm_api_key, config.external_llm_base_url)
    external_llm_model = external_llm_models[0].split("/")[-1] if len(external_llm_models) > 0 else None
    handler = AgentHandler(agent_1, agent_2)
    stream = Stream(
        handler=handler,
        track_constraints = {
            "echoCancellation": False,
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            "sampleRate": {"ideal": handler.input_sample_rate},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        },
        modality="audio", 
        mode="send-receive",
        additional_inputs=[
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
        additional_outputs=[
            gr.Textbox(label="Realtime Factor"),
        ],
        additional_outputs_handler=display_handler,
    )
    stream.ui.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Realtime Codec Agent Self Play with FastRTC.")
    add_common_inference_args(parser)

    args = parser.parse_args()

    main(args)