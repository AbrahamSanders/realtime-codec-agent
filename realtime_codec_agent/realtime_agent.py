import numpy as np
import torch
import time
from typing import Tuple, Union
from openai import OpenAI
from xcodec2.modeling_xcodec2 import XCodec2Model
from codec_bpe import codes_to_chars, chars_to_codes, UNICODE_OFFSET_LARGE

from .utils.vllm_utils import get_vllm_modelname

class RealtimeAgentResources:
    def __init__(
        self, 
        vllm_base_url: str = "http://localhost:8000/v1", 
        xcodec2_model: str = "HKUSTAudio/xcodec2", 
        device: Union[str, torch.device] = None,
    ):
        vllm_api_key = "Empty"
        self.client = OpenAI(
            api_key=vllm_api_key,
            base_url=vllm_base_url,
        )
        self.model_name = get_vllm_modelname(vllm_base_url, vllm_api_key)
        if self.model_name is None:
            raise ValueError("Could not find a model hosted by the vLLM server.")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.codec_model = XCodec2Model.from_pretrained(xcodec2_model).eval().to(device)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)

class RealtimeAgentConfig:
    def __init__(
        self, 
        user_voice_enrollment: np.ndarray = None,
        user_voice_enrollment_text: str = None,
        user_identity: str = "A",
        agent_identity: str = "B",
        text_first_temperature: float = 0.8,
        text_first_presence_penalty=1.0,
        text_first_frequency_penalty=1.0,
        audio_first_cont_temperature: float = 0.6, 
        audio_first_trans_temperature: float = 0.2,
        codec_sample_rate: int = 16000,
        codec_framerate: float = 50.0, 
        codec_context_secs: float = 3.0,
        chunk_size_secs: float = 0.3,
        seed: int = 42,
        audio_first_token: str = "<|audio_first|>",
        text_first_token: str = "<|text_first|>",
        header_speaker_token: str = "<|speaker|>",
        end_header_token: str = "<|end_header|>",
        start_audio_token: str = "<|audio|>",
        end_audio_token: str = "<|end_audio|>",
    ):
        self.user_voice_enrollment = user_voice_enrollment
        self.user_voice_enrollment_text = user_voice_enrollment_text
        self.user_identity = user_identity
        self.agent_identity = agent_identity
        self.text_first_temperature = text_first_temperature
        self.text_first_presence_penalty = text_first_presence_penalty
        self.text_first_frequency_penalty = text_first_frequency_penalty
        self.audio_first_cont_temperature = audio_first_cont_temperature
        self.audio_first_trans_temperature = audio_first_trans_temperature
        self.codec_sample_rate = codec_sample_rate
        self.codec_framerate = codec_framerate
        self.codec_context_secs = codec_context_secs
        self.chunk_size_secs = chunk_size_secs
        self.seed = seed
        self.audio_first_token = audio_first_token
        self.text_first_token = text_first_token
        self.header_speaker_token = header_speaker_token
        self.end_header_token = end_header_token
        self.start_audio_token = start_audio_token
        self.end_audio_token = end_audio_token

        # computed
        self.codec_context_samples = int(codec_context_secs * codec_sample_rate)
        self.codec_context_frames = int(codec_context_secs * codec_framerate)
        self.chunk_size_samples = int(chunk_size_secs * codec_sample_rate)
        self.chunk_size_frames = int(chunk_size_secs * codec_framerate)


    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class RealtimeAgent:
    def __init__(self, resources: RealtimeAgentResources = None, config: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        self.resources = resources

        if config is None:
            config = RealtimeAgentConfig()
        self.config = config 

        self.reset()

    def reset(self):
        c = self.config
        common_header = f"{c.header_speaker_token}{c.user_identity}{c.header_speaker_token}{c.agent_identity}{c.end_header_token}"
        self.audio_first_sequence = f"{c.audio_first_token}{common_header}"
        self.text_first_sequence = f"{c.text_first_token}{common_header}"
        if self.config.user_voice_enrollment is not None:
            enrollment_str = self.tokenize_audio(self.config.user_voice_enrollment)
            self.audio_first_sequence += f"{c.start_audio_token}{enrollment_str}{c.end_audio_token} {c.user_identity}: {c.user_voice_enrollment_text}{c.start_audio_token}"
            self.text_first_sequence +=  f" {c.user_identity}: {c.user_voice_enrollment_text}{c.start_audio_token}{enrollment_str}"
            self.audio_history_str = enrollment_str
        else:
            self.audio_first_sequence += c.start_audio_token
            self.text_first_sequence += c.start_audio_token
            self.audio_history_str = ""
        self.audio_history = np.zeros((2, c.chunk_size_samples), dtype=np.int16)
        

    def set_config(self, config: RealtimeAgentConfig):
        self.config = config

    def tokenize_audio(self, audio):
        audio = audio.astype("float32") / 32768.0
        audio = torch.tensor(audio).unsqueeze(0).to(self.resources.device)

        # get audio codes
        encoder_outputs = self.resources.codec_model.encode_code(audio, sample_rate=self.config.codec_sample_rate)

        # convert to unicode string
        audio_codes_str = codes_to_chars(
            encoder_outputs[0],
            65536,
            unicode_offset=UNICODE_OFFSET_LARGE,
        )

        return audio_codes_str

    def detokenize_audio(self, audio_codes_str):
        # convert unicode string to audio codes
        audio_codes, _, end_hanging = chars_to_codes(
            audio_codes_str, 
            1, 
            65536,
            return_hanging_codes_chars=True, 
            return_tensors="pt",
            unicode_offset=UNICODE_OFFSET_LARGE,
        )
        audio_codes = audio_codes.unsqueeze(0).to(self.resources.device)
        
        # decode audio codes with codec
        with torch.no_grad():
            output_audio = self.resources.codec_model.decode_code(audio_codes)

        # return audio
        output_audio = output_audio[0, 0].cpu().numpy()
        output_audio = (output_audio * 32767.0).astype(np.int16)
        return output_audio, end_hanging

    def get_completion(
        self, 
        sequence: str, 
        max_tokens: int, 
        temperature: float, 
        presence_penalty: float = None, 
        frequency_penalty: float = None, 
        stop=None,
    ) -> Tuple[str, str]:
        extra_body = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        completion = self.resources.client.completions.create(
            model=self.resources.model_name, 
            prompt=sequence, 
            seed=self.config.seed if self.config.seed else None,
            max_tokens=max_tokens, 
            temperature=temperature,
            top_p=1.0,
            stream=False,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            extra_body=extra_body,
        )
        completion_text = completion.choices[0].text
        finish_reason = completion.choices[0].finish_reason
        return completion_text, finish_reason

    def predict_transcription(self):
        # decide if it's time to transcribe by generating the next chunk in the audio-first direction and checking if
        # the model predicts a transcription within it
        _, finish_reason = self.get_completion(
            self.audio_first_sequence,
            max_tokens=self.config.chunk_size_frames,
            temperature=self.config.audio_first_cont_temperature,
            stop=self.config.end_audio_token,
        )
        if finish_reason == "stop":
            # predict the transcription
            self.audio_first_sequence += self.config.end_audio_token
            transcription, _ = self.get_completion(
                self.audio_first_sequence,
                max_tokens=100,
                temperature=self.config.audio_first_trans_temperature,
                stop=self.config.start_audio_token,
            )
            # insert into text-first sequence if transcription belongs to the user
            if transcription.lstrip().startswith(self.config.user_identity):
                # identify the first chunk in the audio sequence that was transcribed
                last_audio_start_pos = self.audio_first_sequence.rfind(self.config.start_audio_token) + len(self.config.start_audio_token)
                first_chunk = self.audio_first_sequence[last_audio_start_pos:last_audio_start_pos+self.config.chunk_size_frames]
                # locate the chunk in the text-first sequence
                first_chunk_pos = self.text_first_sequence.rfind(first_chunk)
                if first_chunk_pos != -1:
                    # insert it here
                    self.text_first_sequence = (
                        f"{self.text_first_sequence[:first_chunk_pos]}{self.config.end_audio_token}"
                        f"{transcription}"
                        f"{self.config.start_audio_token}{self.text_first_sequence[first_chunk_pos:]}"
                    )
                else:
                    print("Warning: could not find the chunk in the text-first sequence")
            # append to the audio-first sequence
            self.audio_first_sequence += f"{transcription}{self.config.start_audio_token}"

    def predict_next_chunk(self):
        audio_completion_text, finish_reason = self.get_completion(
            self.text_first_sequence,
            max_tokens=self.config.chunk_size_frames,
            temperature=self.config.text_first_temperature,
            presence_penalty=self.config.text_first_presence_penalty,
            frequency_penalty=self.config.text_first_frequency_penalty,
            stop=self.config.end_audio_token,
        )
        self.audio_first_sequence += audio_completion_text
        self.text_first_sequence += audio_completion_text
        self.audio_history_str += audio_completion_text
        if finish_reason == "stop":
            # get the utterance text completion from the model
            self.text_first_sequence += f"{self.config.end_audio_token} {self.config.agent_identity}:"
            speaker_text, _ = self.get_completion(
                self.text_first_sequence,
                max_tokens=100,
                temperature=self.config.text_first_temperature,
                presence_penalty=self.config.text_first_presence_penalty,
                frequency_penalty=self.config.text_first_frequency_penalty,
                stop=self.config.start_audio_token,
            )
            self.text_first_sequence += f"{speaker_text}{self.config.start_audio_token}"
            remaining_frames = self.config.chunk_size_frames - len(audio_completion_text)
            audio_completion_text2, _ = self.get_completion(
                self.text_first_sequence,
                max_tokens=remaining_frames,
                temperature=self.config.text_first_temperature,
                presence_penalty=self.config.text_first_presence_penalty,
                frequency_penalty=self.config.text_first_frequency_penalty,
            )
            self.audio_first_sequence += audio_completion_text2
            self.text_first_sequence += audio_completion_text2
            self.audio_history_str += audio_completion_text2

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        print(f'Data from microphone:{audio_data.shape, audio_data.dtype, audio_data.min(), audio_data.max()}')
        if audio_data.shape[-1] != self.config.chunk_size_samples:
            raise ValueError(f"audio_data must have length {self.config.chunk_size_samples}, but got {audio_data.shape[-1]}")

        # insert the input audio to the audio history at the current chunk
        self.audio_history[0, -self.config.chunk_size_samples:] = audio_data

        # tokenize the input audio chunk with context
        if audio_data.max() > 200.0:
            # zero out the predicted audio chunk
            self.audio_history[1, -self.config.chunk_size_samples:] = np.zeros_like(audio_data)
            audio_with_ctx = self.audio_history[..., -self.config.codec_context_samples:].astype(np.float32) / 32767.0
            audio_with_ctx = audio_with_ctx.mean(axis=0)
            audio_with_ctx = (audio_with_ctx * 32767.0).astype(np.int16)

            input_audio_str = self.tokenize_audio(audio_with_ctx)
            input_audio_str = input_audio_str[-self.config.chunk_size_frames:]
            # Replace the predicted audio chunk codes with the input audio codes
            # TODO: this could end up clipping off the last text prediction
            self.audio_first_sequence = self.audio_first_sequence[:-self.config.chunk_size_frames] + input_audio_str
            self.text_first_sequence = self.text_first_sequence[:-self.config.chunk_size_frames] + input_audio_str
            self.audio_history_str = self.audio_history_str[:-self.config.chunk_size_frames] + input_audio_str

        # get the audio transcription from the model up to the current chunk, if predicted
        self.predict_transcription()

        # get the audio completion from the model for the next chunk
        self.predict_next_chunk()

        # detokenize the audio completion with context
        audio_str_with_context = self.audio_history_str[-self.config.codec_context_frames:]
        processed_audio, _ = self.detokenize_audio(audio_str_with_context)
        processed_audio = processed_audio[..., -self.config.chunk_size_samples:]

        # append the output audio to the audio history as the next chunk with a silent placeholder for the next input
        next_chunk_hist = np.stack((np.zeros_like(processed_audio), processed_audio), axis=0)
        self.audio_history = np.concatenate((self.audio_history, next_chunk_hist), axis=-1)

        print(f'Data from model:{processed_audio.shape, processed_audio.dtype, processed_audio.min(), processed_audio.max()}')
        return processed_audio

class RealtimeAgentMultiprocessing:
    def __init__(
        self, 
        wait_until_running: bool = True,
        config: RealtimeAgentConfig = None,
        vllm_base_url: str = "http://localhost:8000/v1", 
        xcodec2_model: str = "HKUSTAudio/xcodec2", 
        device: Union[str, torch.device] = None,
    ):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.sequence_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)
        self.paused = ctx.Value(c_bool, True)
        self.reset_flag = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(
            target=self.execute, 
            daemon=True, 
            args=(config, vllm_base_url, xcodec2_model, device),
        )
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            time.sleep(0.01)

    def is_running(self):
        return self.running.value
    
    def is_paused(self):
        return self.paused.value

    def execute(self, config: RealtimeAgentConfig, vllm_base_url: str, xcodec2_model: str, device: Union[str, torch.device]):
        agent_resources = RealtimeAgentResources(
            vllm_base_url=vllm_base_url, 
            xcodec2_model=xcodec2_model, 
            device=device,
        )
        agent = RealtimeAgent(resources=agent_resources, config=config)

        self.running.value = True
        print(">>> Agent is running! <<<")
        while True:
            try:
                new_config = self._skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config
                    agent.set_config(config)
                    self.reset()
                    print(">>> Config updated! <<<")

                if self.reset_flag.value:
                    agent.reset()
                    self.reset_flag.value = False
                    print(">>> Agent reset! <<<")

                if not self.is_paused() and not self.input_queue.empty():
                    input_audio = self.input_queue.get()
                    output_audio = agent.process_audio(input_audio)
                    self.output_queue.put(output_audio)

                    self.sequence_queue.put((
                        agent.audio_first_sequence,
                        agent.text_first_sequence,
                        agent.audio_history,
                    ))

            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            
            if self.is_paused():
                time.sleep(0.05)

    def _skip_queue(self, queue):
        val = None
        while not queue.empty():
            val = queue.get()
        return val

    def reset(self):
        self.reset_flag.value = True

    def pause(self):
        self.paused.value = True
        print(">>> Agent paused! <<<")

    def resume(self):
        self.paused.value = False
        print(">>> Agent resumed! <<<")

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        if self.output_queue.empty():
            return None
        return self.output_queue.get()
    
    def next_sequence(self):
        return self._skip_queue(self.sequence_queue)