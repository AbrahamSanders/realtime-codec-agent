import numpy as np
import torch
import time
import itertools
from typing import Tuple, Union, List
from openai import OpenAI

from .audio_tokenizer import AudioTokenizer
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
        
        self.audio_tokenizer = AudioTokenizer(codec_model=xcodec2_model, device=device)

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
        text_first_presence_penalty=0.5,
        text_first_frequency_penalty=0.5,
        audio_first_cont_temperature: float = 0.6, 
        audio_first_trans_temperature: float = 0.2,
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
        self.chunk_size_secs = chunk_size_secs
        self.seed = seed
        self.audio_first_token = audio_first_token
        self.text_first_token = text_first_token
        self.header_speaker_token = header_speaker_token
        self.end_header_token = end_header_token
        self.start_audio_token = start_audio_token
        self.end_audio_token = end_audio_token

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class RealtimeAgent:
    def __init__(self, resources: RealtimeAgentResources = None, config: RealtimeAgentConfig = None):
        if resources is None:
            resources = RealtimeAgentResources()
        self.resources = resources

        if config is None:
            config = RealtimeAgentConfig()
        self.set_config(config)

        self.reset()

    def reset(self):
        self.resources.audio_tokenizer.reset_context()
        c = self.config
        common_header = f"{c.header_speaker_token}{c.user_identity}{c.header_speaker_token}{c.agent_identity}{c.end_header_token}"
        self.audio_first_sequence = f"{c.audio_first_token}{common_header}"
        self.text_first_sequence = f"{c.text_first_token}{common_header}"
        if self.config.user_voice_enrollment is not None:
            silence_audio_str = self.resources.audio_tokenizer.tokenize_audio(np.zeros_like(self.config.user_voice_enrollment))
            enrollment_audio_str = self.resources.audio_tokenizer.tokenize_audio(self.config.user_voice_enrollment)
            enrollment_audio_str = "".join(list(itertools.chain.from_iterable(zip(*[enrollment_audio_str, silence_audio_str]))))

            self.audio_first_sequence += f"{c.start_audio_token}{enrollment_audio_str}{c.end_audio_token} {c.user_identity}: {c.user_voice_enrollment_text}{c.start_audio_token}"
            self.text_first_sequence +=  f" {c.user_identity}: {c.user_voice_enrollment_text}{c.start_audio_token}{enrollment_audio_str}"
        else:
            self.audio_first_sequence += c.start_audio_token
            self.text_first_sequence += c.start_audio_token
        self.audio_history = np.zeros((2, 0), dtype=np.int16)

    def set_config(self, config: RealtimeAgentConfig):
        self.config = config
        self.chunk_size_samples = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.sampling_rate)
        self.chunk_size_frames = int(self.config.chunk_size_secs * self.resources.audio_tokenizer.framerate * 2)

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
        completion_str = completion.choices[0].text
        finish_reason = completion.choices[0].finish_reason
        return completion_str, finish_reason

    def predict_transcription(self):
        # decide if it's time to transcribe by generating the next chunk in the audio-first direction and checking if
        # the model predicts a transcription within it
        _, finish_reason = self.get_completion(
            self.audio_first_sequence,
            max_tokens=self.chunk_size_frames,
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
                first_chunk = self.audio_first_sequence[last_audio_start_pos:last_audio_start_pos+self.chunk_size_frames]
                # locate the chunk in the text-first sequence
                first_chunk_pos = self.text_first_sequence.rfind(first_chunk)
                # insert it here
                if first_chunk_pos != -1:
                    start_audio_token = self.config.start_audio_token
                    end_audio_token = self.config.end_audio_token
                    if self.text_first_sequence[first_chunk_pos-len(self.config.start_audio_token):first_chunk_pos] == self.config.start_audio_token:
                        first_chunk_pos -= len(self.config.start_audio_token)
                        start_audio_token = end_audio_token = ""
                    self.text_first_sequence = (
                        f"{self.text_first_sequence[:first_chunk_pos]}{end_audio_token}"
                        f"{transcription}"
                        f"{start_audio_token}{self.text_first_sequence[first_chunk_pos:]}"
                    )
                else:
                    print("Warning: could not find the chunk in the text-first sequence")
            # append to the audio-first sequence
            self.audio_first_sequence += f"{transcription}{self.config.start_audio_token}"

    def predict_next_chunk(self) -> List[str]:
        completion_prompt = self.text_first_sequence
        completion_segments = []
        remaining_frames = self.chunk_size_frames
        while True:
            completion_str, _ = self.get_completion(
                completion_prompt,
                max_tokens=remaining_frames,
                temperature=self.config.text_first_temperature,
                presence_penalty=self.config.text_first_presence_penalty,
                frequency_penalty=self.config.text_first_frequency_penalty,
                stop=self.config.end_audio_token,
            )
            # make sure completion length is even (ends with a complete two-channel frame)
            div_rem = len(completion_str) % 2
            completion_str = completion_str[:(-div_rem or None)]
            completion_segments.append(completion_str)
            # check if the completion is finished
            remaining_frames -= len(completion_str)
            if remaining_frames == 0:
                break
            # get the utterance text completion from the model
            completion_prompt += completion_str
            segment_start_pos = len(completion_prompt)
            completion_prompt += f"{self.config.end_audio_token} {self.config.agent_identity}:"
            speaker_text, _ = self.get_completion(
                completion_prompt,
                max_tokens=100,
                temperature=self.config.text_first_temperature,
                presence_penalty=self.config.text_first_presence_penalty,
                frequency_penalty=self.config.text_first_frequency_penalty,
                stop=self.config.start_audio_token,
            )
            completion_prompt += f"{speaker_text}{self.config.start_audio_token}"
            completion_segments.append(completion_prompt[segment_start_pos:])

        return completion_segments

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        print(f'Data from microphone:{audio_data.shape, audio_data.dtype, audio_data.min(), audio_data.max()}')
        if audio_data.shape[-1] != self.chunk_size_samples:
            raise ValueError(f"audio_data must have length {self.chunk_size_samples}, but got {audio_data.shape[-1]}")

        if np.abs(audio_data).max() < 100.0:
            audio_data = np.zeros_like(audio_data)

        # get the audio transcription from the model up to the current chunk, if predicted
        self.predict_transcription()

        # get the audio completion from the model for the next chunk. This comes back as a list of alternating "segments" of
        # audio and text / control tokens, where the length of all audio segments sum up to self.chunk_size_frames. 
        # For example: "audio_codes_1<|end_audio|> B: what's up?<|audio|>audio_codes_2"
        # would come back as: ["audio_codes_1", "<|end_audio|> B: what's up?<|audio|>", "audio_codes_2"]
        completion_segments = self.predict_next_chunk()

        # tokenize the audio input
        input_audio_str = self.resources.audio_tokenizer.tokenize_audio(audio_data)

        # substitute the actual input (user) audio codes in place of the predicted ones, and append the completion to both sequences.
        # text completions only get appended to the text-first sequence while audio completions get appended to both sequences.
        audio_str_ch_2 = ""
        for i, segment in enumerate(completion_segments):
            # even segments are audio, odd segments are text + control tokens
            if i % 2 == 0:
                # split audio into two channels, discarding the first (user) channel.
                # predicted user audio may be used for planning in the future but for now we just discard it
                seg_audio_str_ch_2 = segment[1::2]
                # interleave using the actual user input audio codes
                ch_1_start = len(audio_str_ch_2)
                seg_audio_str_ch_1 = input_audio_str[ch_1_start:ch_1_start+len(seg_audio_str_ch_2)]
                seg_audio_str_interleaved = "".join(list(itertools.chain.from_iterable(zip(*[seg_audio_str_ch_1, seg_audio_str_ch_2]))))
                # append the interleaved audio codes to both sequences
                self.audio_first_sequence += seg_audio_str_interleaved
                self.text_first_sequence += seg_audio_str_interleaved
                audio_str_ch_2 += seg_audio_str_ch_2
            else:
                # append the text + control tokens segment to the text-first sequence
                self.text_first_sequence += segment

        # detokenize the output (agent) audio from channel 2
        (_, audio_ch_2), _ = self.resources.audio_tokenizer.detokenize_audio(audio_str_ch_2)
        audio_ch_2 = (audio_ch_2 * 32767.0).astype(np.int16)

        # append the input and output audio to the audio history
        self.audio_history = np.concatenate((self.audio_history, np.stack([audio_data, audio_ch_2], axis=0)), axis=-1)

        print(f'Data from model:{audio_ch_2.shape, audio_ch_2.dtype, audio_ch_2.min(), audio_ch_2.max()}')
        return audio_ch_2

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
                    self._skip_queue(self.input_queue)
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