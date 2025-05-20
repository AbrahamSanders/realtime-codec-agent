import numpy as np
import argparse
import torch
import torchaudio
import av
import streamlit as st
from scipy.io.wavfile import read
from typing import List
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from realtime_codec_agent.realtime_agent import RealtimeAgentMultiprocessing, RealtimeAgentConfig

@st.cache_resource
def get_agent(_args):
    agent = RealtimeAgentMultiprocessing(
        config=RealtimeAgentConfig(chunk_size_secs=BLOCK_SIZE/SAMPLE_RATE),
        vllm_base_url=_args.vllm_base_url,
    )
    return agent

class AudioClient:
    def __init__(self, agent):
        self.agent = agent
    
        self.sound_check = False
        self.downsampler = torchaudio.transforms.Resample(STREAMING_SAMPLE_RATE, SAMPLE_RATE)
        self.upsampler = torchaudio.transforms.Resample(SAMPLE_RATE, STREAMING_SAMPLE_RATE)
        self.in_buffer = None
        self.out_buffer = None
        self.sequence_data = None

    def update_agent_config(
        self, 
        user_voice_enrollment: bytes,
        user_voice_enrollment_text: str,
        text_first_temperature: float, 
        text_first_presence_penalty: float,
        text_first_frequency_penalty: float,
        audio_first_cont_temperature: float, 
        audio_first_trans_temperature: float,
    ):
        if user_voice_enrollment is not None:
            sr, user_voice_enrollment = read(user_voice_enrollment)
            #user_voice_enrollment = self.downsample(user_voice_enrollment)
        config = RealtimeAgentConfig(
            user_voice_enrollment=user_voice_enrollment,
            user_voice_enrollment_text=user_voice_enrollment_text,
            text_first_temperature=text_first_temperature,
            text_first_presence_penalty=text_first_presence_penalty,
            text_first_frequency_penalty=text_first_frequency_penalty,
            audio_first_cont_temperature=audio_first_cont_temperature,
            audio_first_trans_temperature=audio_first_trans_temperature,
            chunk_size_secs=BLOCK_SIZE/SAMPLE_RATE,
        )
        self.agent.queue_config(config)

    def stop(self): 
        self.in_buffer = None
        self.out_buffer = None
        self.agent.pause()
        self.agent.reset()
    
    def _resample(self, audio_data: np.ndarray, resampler: torchaudio.transforms.Resample) -> np.ndarray:
        audio_data = audio_data.astype(np.float32) / 32768.0
        audio_data = resampler(torch.tensor(audio_data)).numpy()
        audio_data = (audio_data * 32767.0).astype(np.int16)
        return audio_data
    
    def upsample(self, audio_data: np.ndarray) -> np.ndarray:
        return self._resample(audio_data, self.upsampler)
    
    def downsample(self, audio_data: np.ndarray) -> np.ndarray:
        return self._resample(audio_data, self.downsampler)
    
    def from_s16_format(self, audio_data: np.ndarray, channels: int) -> np.ndarray:
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2).T
        else:
            audio_data = audio_data.reshape(-1)
        return audio_data
    
    def to_s16_format(self, audio_data: np.ndarray):
        if len(audio_data.shape) == 2 and audio_data.shape[0] == 2:
            audio_data = audio_data.T.reshape(1, -1)
        elif len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
        return audio_data
    
    def to_channels(self, audio_data: np.ndarray, channels: int) -> np.ndarray:
        current_channels = audio_data.shape[0] if len(audio_data.shape) == 2 else 1
        if current_channels == channels:
            return audio_data
        elif current_channels == 1 and channels == 2:
            audio_data = np.tile(audio_data, 2).reshape(2, -1)
        elif current_channels == 2 and channels == 1:
            audio_data = audio_data.astype(np.float32) / 32768.0
            audio_data = audio_data.mean(axis=0)
            audio_data = (audio_data * 32767.0).astype(np.int16)
        return audio_data
    
    async def queued_audio_frames_callback(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
        out_frames = []
        for frame in frames:
            # Read in audio
            audio_data = frame.to_ndarray()

            # Convert input audio from s16 format, convert to `CHANNELS` number of channels, and downsample
            audio_data = self.from_s16_format(audio_data, len(frame.layout.channels))
            audio_data = self.to_channels(audio_data, CHANNELS)
            audio_data = self.downsample(audio_data)

            # Add audio to input buffer
            if self.in_buffer is None:
                self.in_buffer = audio_data
            else:
                self.in_buffer = np.concatenate((self.in_buffer, audio_data), axis=-1)
            
            # Take BLOCK_SIZE samples from input buffer if available for processing
            if self.in_buffer.shape[0] >= BLOCK_SIZE:
                audio_data = self.in_buffer[:BLOCK_SIZE]
                self.in_buffer = self.in_buffer[BLOCK_SIZE:]
            else:
                audio_data = None
            
            # Process audio if available
            if not self.sound_check:
                if self.agent.is_paused():
                    self.agent.resume()
                if audio_data is not None:
                    self.agent.queue_input(audio_data)
                audio_data = self.agent.next_output()
                sequence_data = self.agent.next_sequence()
                if sequence_data is not None:
                    self.sequence_data = sequence_data
            elif not self.agent.is_paused():
                self.agent.pause()

            # add resulting audio to output buffer     
            if audio_data is not None:
                if self.out_buffer is None:
                    self.out_buffer = audio_data
                else:
                    self.out_buffer = np.concatenate((self.out_buffer, audio_data), axis=-1)

            # Take `out_samples` samples from output buffer if available for output
            out_samples = int(frame.samples * SAMPLE_RATE / STREAMING_SAMPLE_RATE)
            if self.out_buffer is not None and self.out_buffer.shape[0] >= out_samples:
                audio_data = self.out_buffer[:out_samples]
                self.out_buffer = self.out_buffer[out_samples:]
            else:
                audio_data = None

            # Output silence if no audio data available
            if audio_data is None:
                # output silence
                audio_data = np.zeros(out_samples, dtype=np.int16)
            
            # Upsample output audio, convert to original number of channels, and convert to s16 format
            audio_data = self.upsample(audio_data)
            audio_data = self.to_channels(audio_data, len(frame.layout.channels))
            audio_data = self.to_s16_format(audio_data)

            # return audio data as AudioFrame
            new_frame = av.AudioFrame.from_ndarray(audio_data, format=frame.format.name, layout=frame.layout.name)
            new_frame.sample_rate = frame.sample_rate
            out_frames.append(new_frame)

        return out_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Client')
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--use_ice_servers", action="store_true", help="Use public STUN servers")
    
    args = parser.parse_args()
    
    # Audio settings
    STREAMING_SAMPLE_RATE = 48000
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 4800 # 0.3 secs
    CHANNELS = 1
    
    st.title("full-duplex webrtc demo!")
    st.markdown("""
    Welcome to the audio processing interface! Here you can talk live with the full-duplex model.
    
    To begin, click the START button below and allow microphone access.
    """)

    audio_client = st.session_state.get("audio_client")
    if audio_client is None:
        agent = get_agent(args)
        audio_client = AudioClient(agent)
        st.session_state.audio_client = audio_client

    with st.sidebar:
        st.markdown("## Inference Settings")
        user_voice_enrollment = st.audio_input("Record yourself saying something.")
        user_voice_enrollment_text = st.text_input("What did you say?", "hello, how are you?")
        text_first_temperature = st.slider("Text First Temperature", 0.0, 1.0, 0.8, 0.01)
        text_first_presence_penalty = st.slider("Text First Presence Penalty", -2.0, 2.0, 0.5, 0.1)
        text_first_frequency_penalty = st.slider("Text First Frequency Penalty", -2.0, 2.0, 0.5, 0.1)
        audio_first_cont_temperature = st.slider("Audio First Continuation Temperature", 0.0, 1.0, 0.6, 0.01)
        audio_first_trans_temperature = st.slider("Audio First Transcription Temperature", 0.0, 1.0, 0.2, 0.01)
        if st.button("Update"):
            audio_client.update_agent_config(
                user_voice_enrollment=user_voice_enrollment,
                user_voice_enrollment_text=user_voice_enrollment_text,
                text_first_temperature=text_first_temperature,
                text_first_presence_penalty=text_first_presence_penalty,
                text_first_frequency_penalty=text_first_frequency_penalty,
                audio_first_cont_temperature=audio_first_cont_temperature,
                audio_first_trans_temperature=audio_first_trans_temperature,
            )
            st.write("Updated agent config!")

        st.markdown("## Microphone Settings")
        audio_client.sound_check = st.toggle("Sound Check (Echo)", value=False)
        echo_cancellation = st.toggle("Echo Cancellation*‡", value=False)
        noise_suppression = st.toggle("Noise Suppression*", value=False)
        st.markdown(r"\* *Restart stream to take effect*")
        st.markdown("‡ *May cause audio to cut out*")

    # Use a free STUN server from Google if --use_ice_servers is given
    # (found in get_ice_servers() at https://github.com/whitphx/streamlit-webrtc/blob/main/sample_utils/turn.py)
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} if args.use_ice_servers else None
    audio_config = {"echoCancellation": echo_cancellation, "noiseSuppression": noise_suppression}
    webrtc_streamer(
        key="streamer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"audio": audio_config, "video": False},
        queued_audio_frames_callback=audio_client.queued_audio_frames_callback,
        on_audio_ended=audio_client.stop,
        async_processing=True,
    )
    if audio_client.sequence_data is not None:
        audio_first_sequence, text_first_sequence, audio_history = audio_client.sequence_data

        st.markdown("### Audio First Sequence")
        st.markdown(audio_first_sequence)

        st.markdown("### Text First Sequence")
        st.markdown(text_first_sequence)

        st.markdown("### Audio History")
        st.audio(audio_history, sample_rate=SAMPLE_RATE)
