from typing import Tuple, Optional
import io
import base64
import requests
import soundfile as sf
import numpy as np

class ExternalTTSClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8001", chunk_size_secs: float = 0.1):
        self.server_url = server_url.rstrip("/")
        self.session_id = "default_session"
        self.chunk_size_secs = chunk_size_secs
        self.stream_resp = None
        self.stream = None

    def _encode_audio_numpy_to_base64(self, audio_input: Tuple[int, np.ndarray]) -> str:
        sample_rate, data = audio_input
        buf = io.BytesIO()
        sf.write(buf, data, sample_rate, format="WAV")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
    def set_voice_enrollment(self, voice_enrollment: Optional[Tuple[int, np.ndarray]] = None, prompt_text: Optional[str] = None) -> None:
        wav_base64 = None
        if voice_enrollment is not None:
            wav_base64 = self._encode_audio_numpy_to_base64(voice_enrollment)
        with requests.post(
            f"{self.server_url}/set_voice_enrollment",
            json={
                "session_id": self.session_id,
                "wav_base64": wav_base64,
                "prompt_text": prompt_text,
            },
        ) as enroll_resp:
            enroll_resp.raise_for_status()

    def prep_stream(self, text: str) -> None:
        try:
            self.close_stream()
            self.stream_resp = requests.post(
                f"{self.server_url}/stream",
                json={
                    "session_id": self.session_id,
                    "text": text,
                    "chunk_size_secs": self.chunk_size_secs,
                },
                stream=True,
            )
            self.stream_resp.raise_for_status()
            self.stream = self.stream_resp.iter_lines(decode_unicode=True)
        except Exception as e:
            try:
                self.close_stream()
            except Exception:
                pass
            raise e

    def next_chunk(self) -> Optional[str]:
        if self.stream is None:
            return None
        try:
            chunk = next(self.stream, None)
            if chunk is None:
                self.close_stream()
            return chunk
        except Exception as e:
            try:
                self.close_stream()
            except Exception:
                pass
            raise e
        
    def close_stream(self) -> None:
        if self.stream_resp is not None:
            self.stream_resp.close()
            self.stream_resp = None
            self.stream = None
        