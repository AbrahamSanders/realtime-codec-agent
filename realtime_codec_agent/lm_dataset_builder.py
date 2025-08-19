from typing import Optional, Union, Iterator, List, Tuple, Dict
from tqdm import tqdm
from enum import Enum
import numpy as np
import itertools
import re
import random
import os

from codec_bpe.core.converter import codes_to_chars, UNICODE_OFFSET
from codec_bpe.core.utils import get_codes_files

from .utils.transcript_utils import load_transcript, set_agent_speaker, is_speaker_channel_isolated

class InterleaveOrder(Enum):
    AUDIO_ONLY = "audio_only"
    TEXT_ONLY = "text_only"
    AUDIO_FIRST = "audio_first"
    TEXT_FIRST = "text_first"
    AGENT = "agent"
    ALL = "all"

class LMDatasetBuilder:
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: float,
        interleave_order: InterleaveOrder = InterleaveOrder.ALL,
        audio_start_token: str = "<|audio|>",
        audio_end_token: str = "<|end_audio|>",
        header_audio_only_token: str = "<|audio_only|>",
        header_text_only_token: str = "<|text_only|>",
        header_audio_first_token: str = "<|audio_first|>",
        header_text_first_token: str = "<|text_first|>",
        header_agent_token: str = "<|agent|>",
        header_agent_voice_token: str = "<|agent_voice|>",
        header_speaker_token: str = "<|speaker|>",
        header_end_token: str = "<|end_header|>",
        unicode_offset: int = UNICODE_OFFSET,
        context_secs: float = 80.0,
        overlap_secs: float = 20.0,
        text_only_context_words: int = 3000,
        text_only_overlap_words: int = 750,
        max_voice_enrollment_secs: float = 6.0,
        voice_enrollment_selection_seed: int = 42,
        agent_identity: str = "A",
        speaker_proportion_threshold: float = 0.1,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.interleave_order = interleave_order
        self.unicode_offset = unicode_offset
        self.context_secs = context_secs
        self.overlap_secs = overlap_secs
        self.text_only_context_words = text_only_context_words
        self.text_only_overlap_words = text_only_overlap_words
        self.max_voice_enrollment_secs = max_voice_enrollment_secs
        self.voice_enrollment_selection_seed = voice_enrollment_selection_seed
        self.agent_identity = agent_identity
        self.speaker_proportion_threshold = speaker_proportion_threshold

        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.header_audio_only_token = header_audio_only_token
        self.header_text_only_token = header_text_only_token
        self.header_audio_first_token = header_audio_first_token
        self.header_text_first_token = header_text_first_token
        self.header_agent_token = header_agent_token
        self.header_agent_voice_token = header_agent_voice_token
        self.header_speaker_token = header_speaker_token
        self.header_end_token = header_end_token

    def _group_codes_files(self, codes_files: List[str]) -> List[Tuple[str, List[List[str]]]]:
        grouped_codes_files = []
        last_file_root = None
        for codes_file in codes_files:
            codes_file_info = re.match(r"(.+)_c(\d+)[_.]", codes_file)
            if not codes_file_info:
                raise ValueError(
                    f"Invalid codes file name format: {codes_file}. Expected format: *_c<channel>.npy or *_c<channel>_<timestamp>.npy"
                )
            file_root, channel = codes_file_info.group(1), int(codes_file_info.group(2))
            if file_root != last_file_root:
                grouped_codes_files.append((file_root, []))
                last_file_root = file_root
            grouped_codes_files[-1][1].append((codes_file, channel))

        # separate the files in each groups by channel
        channel_grouped_codes_files = []
        for file_root, file_group in grouped_codes_files:
            num_channels = max([channel for _, channel in file_group]) + 1
            channel_grouped_codes_files.append(
                (
                    file_root, 
                    [[f[0] for f in file_group if f[1] == c] for c in range(num_channels)],
                )
            )

        return channel_grouped_codes_files

    def _build_codes_strs(
        self, 
        channels_chars: List[str], 
        transcript_lines: List[Tuple[float, float, str, str]],
        trans_pos_bounds: Tuple[int, int],
        speakers: List[str], 
        channel_map: Dict[str, int],
    ) -> List[Tuple[str, InterleaveOrder, Optional[str]]]:        
        # add a dummy line to handle any audio beyond the last transcribed line
        transcript_lines = transcript_lines.copy()
        transcript_lines.append((None, None, None, None))

        # build the codes strings
        codes_strs = []
        if self.interleave_order == InterleaveOrder.AUDIO_ONLY or self.interleave_order == InterleaveOrder.ALL:
            # build the audio-only codes string
            codes_str = self._build_codes_str(channels_chars, transcript_lines[-1:], channel_map, InterleaveOrder.AUDIO_ONLY)
            codes_strs.append((codes_str, InterleaveOrder.AUDIO_ONLY, None))
        if  (self.interleave_order == InterleaveOrder.TEXT_ONLY or self.interleave_order == InterleaveOrder.ALL) and len(speakers) > 0:
            # build the text-only codes string
            codes_str = self._build_text_only_str(transcript_lines)
            codes_strs.append((codes_str, InterleaveOrder.TEXT_ONLY, None))
        if (self.interleave_order == InterleaveOrder.AUDIO_FIRST or self.interleave_order == InterleaveOrder.ALL) and len(speakers) > 0:
            # build the audio-first codes string
            codes_str = self._build_codes_str(channels_chars, transcript_lines, channel_map, InterleaveOrder.AUDIO_FIRST, *trans_pos_bounds)
            codes_strs.append((codes_str, InterleaveOrder.AUDIO_FIRST, None))
        if (self.interleave_order == InterleaveOrder.TEXT_FIRST or self.interleave_order == InterleaveOrder.ALL) and len(speakers) > 0:
            # build the text-first codes string
            codes_str = self._build_codes_str(channels_chars, transcript_lines, channel_map, InterleaveOrder.TEXT_FIRST, *trans_pos_bounds)
            codes_strs.append((codes_str, InterleaveOrder.TEXT_FIRST, None))
        if (self.interleave_order == InterleaveOrder.AGENT or self.interleave_order == InterleaveOrder.ALL) and len(speakers) == 2:
            # build the agent codes strings. The agent speaker uses text-first interleave order, while all other speakers use audio-first.
            # we want to give each speaker in the transcript their turn being the agent, to make sure the model learns speaking and listening
            # roles in all possible speaker combinations.
            for agent_speaker in speakers:
                swapped_transcript_lines, swapped_channel_map = set_agent_speaker(
                    transcript_lines, speakers, channel_map, agent_speaker
                )
                # get two versions of the transcript: one with just the agent speaker and one with all other speakers
                agent_transcript_lines = [line for line in swapped_transcript_lines if line[2] == self.agent_identity or line[2] is None]
                other_transcript_lines = [line for line in swapped_transcript_lines if line[2] != self.agent_identity]
                agent_codes_str = self._build_codes_str(
                    channels_chars, agent_transcript_lines, swapped_channel_map, InterleaveOrder.TEXT_FIRST, *trans_pos_bounds
                )
                other_codes_str = self._build_codes_str(
                    channels_chars, other_transcript_lines, swapped_channel_map, InterleaveOrder.AUDIO_FIRST, *trans_pos_bounds
                )
                # merge the two strings. other_codes_str is passed as codes_str_1 so that if any audio-first and text-first texts
                # co-occur at the same position in the audio, the audio-first transcription text will come before the text-first generation text.
                codes_str = self._merge_codes_strs(other_codes_str, agent_codes_str)
                codes_strs.append((codes_str, InterleaveOrder.AGENT, agent_speaker))

        return codes_strs

    def _get_transcript_start_end_pos(
        self, 
        channels_chars: List[str], 
        transcript_lines: List[Tuple[float, float, str, str]], 
    ) -> Tuple[int, int]:
        if not transcript_lines:
            return 0, len(channels_chars[0])
        min_start_secs = min([line[0] for line in transcript_lines])
        max_end_secs = max([line[1] for line in transcript_lines])
        trans_start_pos = int(min_start_secs * self.codec_framerate * self.num_codebooks)
        trans_end_pos = int(max_end_secs * self.codec_framerate * self.num_codebooks)
        return trans_start_pos, trans_end_pos

    def _build_codes_str(
        self, 
        channels_chars: List[str], 
        transcript_lines: List[Tuple[float, float, str, str]], 
        channel_map: Dict[str, int],
        interleave_order: InterleaveOrder,
        trans_start_pos: Optional[int] = None,
        trans_end_pos: Optional[int] = None,
    ) -> str:
        if interleave_order == InterleaveOrder.AGENT or interleave_order == InterleaveOrder.ALL:
            raise ValueError(f"{interleave_order} cannot be passed here.")
        
        # make sure the agent speaker is always on the first channel
        if channel_map.get(self.agent_identity, {"channel": 0})["channel"] != 0:
            agent_channel = channel_map[self.agent_identity]["channel"]
            swapped_channels_chars = []
            for i in range(len(channels_chars)):
                if i == 0:
                    swapped_channels_chars.append(channels_chars[agent_channel])
                elif i == agent_channel:
                    swapped_channels_chars.append(channels_chars[0])
                else:
                    swapped_channels_chars.append(channels_chars[i])
            channels_chars = swapped_channels_chars

        # build the codes string
        str_parts = []
        last_codes_pos = trans_start_pos if trans_start_pos is not None else 0
        last_speaker = None
        for start_secs, end_secs, speaker, text in transcript_lines:
            transcript_start_secs = end_secs if interleave_order == InterleaveOrder.AUDIO_FIRST else start_secs
            if transcript_start_secs is None:
                line_start_pos = trans_end_pos if trans_end_pos is not None else len(channels_chars[0])
            else:
                # Add any audio up to the point that the transcript line starts
                line_start_pos = int(transcript_start_secs * self.codec_framerate * self.num_codebooks)
                # Roll back to the last complete acoustic unit if we land in the middle of one
                line_start_pos -= line_start_pos % self.num_codebooks
            if line_start_pos > last_codes_pos:
                # add audio start token
                str_parts.append(self.audio_start_token)
                audio_part = [chars[last_codes_pos:line_start_pos] for chars in channels_chars]
                audio_part = list(itertools.chain.from_iterable(zip(*audio_part)))
                audio_part = "".join(audio_part)
                str_parts.append(audio_part)
                last_codes_pos = line_start_pos
                # add audio end token
                str_parts.append(self.audio_end_token)
                # reset last speaker since there is preceding audio
                last_speaker = None
            if speaker is not None:
                # Add the transcript line
                if speaker == last_speaker:
                    str_parts.append(f" {text}")
                else:
                    str_parts.append(f" {speaker}: {text}")
                    last_speaker = speaker

        # compile the codes string
        codes_str = "".join(str_parts)
        return codes_str

    def _build_text_only_str(self, transcript_lines: List[Tuple[float, float, str, str]]) -> str:
        str_parts = [f" {speaker}: {text}" for _, _, speaker, text in transcript_lines if speaker is not None and text]
        text_only_str = "".join(str_parts)
        return text_only_str

    def _merge_codes_strs(self, codes_str_1: str, codes_str_2: str) -> str:
        # We assume codes_str_1 and codes_str_2 have the same audio tokens and a non-overlapping set of transcribed speakers.
        _, codes_str_1_audio = self._get_audio_only_codes_str(codes_str_1)
        _, codes_str_2_audio = self._get_audio_only_codes_str(codes_str_2)
        if codes_str_1_audio != codes_str_2_audio:
            raise ValueError("The audio tokens in the two strings do not match.")
        
        merged_codes_str = ""
        i1, i2 = 0, 0
        while i1 < len(codes_str_1) and i2 < len(codes_str_2):
            if codes_str_1[i1] == codes_str_2[i2] and self._is_audio_code(codes_str_1[i1]):
                # If both strings have the same audio token character, append it and proceed to the next position in both strings
                merged_codes_str += codes_str_1[i1]
                i1 += 1
                i2 += 1
            elif not self._is_audio_code(codes_str_1[i1]):
                # If codes_str_1 has a non-audio character, append it to the merged string and proceed to the next position in codes_str_1
                merged_codes_str += codes_str_1[i1]
                i1 += 1
            elif not self._is_audio_code(codes_str_2[i2]):
                # If codes_str_2 has a non-audio character, append it to the merged string and proceed to the next position in codes_str_2
                merged_codes_str += codes_str_2[i2]
                i2 += 1
        if i1 < len(codes_str_1):
            # If there are remaining characters in codes_str_1, append them
            merged_codes_str += codes_str_1[i1:]
        if i2 < len(codes_str_2):
            # If there are remaining characters in codes_str_2, append them
            merged_codes_str += codes_str_2[i2:]
        
        merged_codes_str = merged_codes_str.replace(f"{self.audio_start_token}{self.audio_start_token}", self.audio_start_token)
        merged_codes_str = merged_codes_str.replace(f"{self.audio_end_token}{self.audio_end_token}", self.audio_end_token)
        merged_codes_str = merged_codes_str.replace(f"{self.audio_start_token}{self.audio_end_token}", "")
        # If the merged string starts with an audio start token followed by a non-audio character, remove the leading audio start token
        if merged_codes_str.startswith(self.audio_start_token) and not self._is_audio_code(merged_codes_str[len(self.audio_start_token)]):
            merged_codes_str = merged_codes_str[len(self.audio_start_token):]

        #Sanity check:
        _, merged_codes_audio = self._get_audio_only_codes_str(merged_codes_str)
        if merged_codes_audio != codes_str_1_audio:
            raise ValueError("The merged audio tokens do not match the original audio tokens.")

        return merged_codes_str

    def _get_audio_only_codes_str(self, codes_str: str) -> str:
        codes_str_chars = np.array(list(codes_str))
        codes_str_ords = codes_str_chars.view(np.int32)
        audio_idx = np.where(codes_str_ords >= self.unicode_offset)[0]
        return audio_idx, "".join(codes_str_chars[audio_idx])

    def _is_audio_code(self, char: str) -> bool:
        return ord(char) >= self.unicode_offset

    def _select_agent_voice(
        self, 
        agent_channel_chars: str, 
        example_start_code: int, 
        example_end_code: int, 
        transcript_lines: List[Tuple[float, float, str, str]],
        agent_speaker: str,
        agent_channel_isolated: bool,
        target_min_candidates: int = 20,
        target_min_length_secs: float = 3.0,
    ) -> Optional[str]:
        speech_ranges = [
            (
                int(start_secs * self.codec_framerate * self.num_codebooks), 
                int(end_secs * self.codec_framerate * self.num_codebooks),
                speaker,
                text, # don't really need the text but it helps for debugging
                end_secs - start_secs,
            ) 
            for start_secs, end_secs, speaker, text in transcript_lines
        ]
        # make sure the samples do not overlap with any other speakers (this would make a confusing voice enrollment sample)
        overlap_table = np.zeros(len(agent_channel_chars), dtype=np.int32)
        if not agent_channel_isolated:
            for start_code, end_code, speaker, _, _ in speech_ranges:
                if speaker != agent_speaker:
                    overlap_table[start_code:end_code] += 1
        # We will select a non-overlapped sample of the agent's speech outside the current example range.
        agent_speech_ranges = [
            (start_code, end_code, text, length_secs) for start_code, end_code, speaker, text, length_secs in speech_ranges 
            if speaker == agent_speaker 
            and length_secs <= self.max_voice_enrollment_secs # within the max voice enrollment length
            and overlap_table[start_code:end_code].sum() == 0 # no overlap with other speakers
            and (end_code <= example_start_code or start_code >= example_end_code) # outside the current example range
        ]
        # extract the voice candidate audio code strings
        voice_candidates = [
            (agent_channel_chars[start_code:end_code], text, length_secs) for start_code, end_code, text, length_secs in agent_speech_ranges
        ]
        # take target_min_candidates longest candidates or all that are target_min_length_secs and longer, whichever yields more candidates.
        voice_candidates.sort(key=lambda x: x[2], reverse=True)
        voice_candidate_strs = [
            (candidate_str, text) for i, (candidate_str, text, length_secs) in enumerate(voice_candidates) 
            if i < target_min_candidates or length_secs >= target_min_length_secs
        ]
        # Select a random voice candidate from voice_candidate_strs
        if not voice_candidate_strs:
            return None
        selected_voice = random.choice(voice_candidate_strs)
        return selected_voice[0]

    def _build_common_header(self, interleave_order: InterleaveOrder, speakers: List[str]) -> str:
        header = ""
        if interleave_order == InterleaveOrder.AUDIO_ONLY:
            header += self.header_audio_only_token
        if interleave_order == InterleaveOrder.TEXT_ONLY:
            header += self.header_text_only_token
        if interleave_order == InterleaveOrder.AUDIO_FIRST:
            header += self.header_audio_first_token
        if interleave_order == InterleaveOrder.TEXT_FIRST:
            header += self.header_text_first_token
        if interleave_order == InterleaveOrder.AGENT:
            header += self.header_agent_token
        if interleave_order != InterleaveOrder.AUDIO_ONLY:
            header += "".join([f"{self.header_speaker_token} {speaker}" for speaker in speakers])
        return header

    def iterate_examples(
        self, 
        codes_path: str, 
        transcripts_path: str, 
        codes_filter: Optional[Union[str, List[str]]] = None,
        codes_filter_exclude: Optional[Union[str, List[str]]] = None,
    ) -> Iterator[str]:
        # get the codes files
        codes_files = get_codes_files(codes_path, codes_filter)
        if codes_filter_exclude:
            if isinstance(codes_filter_exclude, str):
                codes_filter_exclude = [codes_filter_exclude]
            codes_files = [f for f in codes_files if not any(ex in f for ex in codes_filter_exclude)]
        # group codes files by root filename (minus channel and starting timestamp) and then by channel
        grouped_codes_files = self._group_codes_files(codes_files)
        
        # iterate over each group of codes files
        for file_root, file_channels in tqdm(grouped_codes_files, desc="Codes file groups"):
            # load the transcript if it exists
            rel_file_root = os.path.relpath(file_root, codes_path)
            transcript_file = os.path.join(transcripts_path, f"{rel_file_root}.txt")
            transcript_lines, speakers, channel_map = load_transcript(transcript_file, self.speaker_proportion_threshold)
            
            if self.interleave_order not in [InterleaveOrder.AUDIO_ONLY, InterleaveOrder.ALL] and not transcript_lines:
                print(f"No transcript found for {file_root}. Skipping file...")
                continue

            num_channels = len(file_channels)
            if num_channels == 1:
                # do not use the transcript's channel map if the audio was encoded as mono
                channel_map = {}
            context_codes = int(self.context_secs * self.codec_framerate * self.num_codebooks * num_channels)
            overlap_codes = int(self.overlap_secs * self.codec_framerate * self.num_codebooks * num_channels)
            if context_codes % (self.num_codebooks * num_channels) != 0 or overlap_codes % (self.num_codebooks * num_channels) != 0:
                raise ValueError(
                    f"context_codes and overlap_codes must be divisible by {self.num_codebooks * num_channels} "
                    "To ensure examples do not start or end in the middle of an acoustic unit or channel pair."
                )
            
            # concatenate all codes files in each group for each channel
            codes = np.stack(
                [
                    np.concatenate([np.load(file) for file in file_group], axis=-1) 
                    for file_group in file_channels
                ], 
                axis=0,
            )
            if len(codes.shape) == 5:
                codes = codes[:, 0, 0]
            elif len(codes.shape) == 4:
                codes = codes[:, 0]
            codes = codes[:, :self.num_codebooks] # shape: (num_channels, num_codebooks, sequence_length)

            # convert codes to unicode string
            channels_chars = [
                codes_to_chars(
                    ch_codes, 
                    self.codebook_size, 
                    copy_before_conversion=False,
                    unicode_offset=self.unicode_offset,
                ) for ch_codes in codes
            ]
            
            # compute transcript start and end positions in channels_chars
            trans_pos_bounds = self._get_transcript_start_end_pos(channels_chars, transcript_lines)

            # build the codes strings (audio unicode characters + additional text tokens)
            codes_strs = self._build_codes_strs(channels_chars, transcript_lines, trans_pos_bounds, speakers, channel_map)

            # build the examples
            random.seed(self.voice_enrollment_selection_seed)
            for codes_str, interleave_order, agent_speaker in codes_strs:
                metadata = {
                    "file_id": rel_file_root,
                    "interleave_order": interleave_order.value,
                    "agent_speaker": agent_speaker,
                    "example_index": 0,
                }
                if interleave_order == InterleaveOrder.TEXT_ONLY:
                    text_only_words = codes_str.split()
                    speaker_words = {f"{speaker}:" for speaker in speakers}
                    start_word = 0
                    while True:
                        end_word = start_word + self.text_only_context_words
                        example = " ".join(text_only_words[start_word:end_word])
                        # build the header
                        header = self._build_common_header(interleave_order, speakers)
                        example = f"{header}{self.header_end_token} {example}"
                        yield example, metadata.copy()
                        metadata["example_index"] += 1
                        if end_word >= len(text_only_words):
                            break
                        start_word = end_word - self.text_only_overlap_words
                        # roll forward until we hit the beginning of the next speaker turn or catch up to the end_word
                        while text_only_words[start_word] not in speaker_words and start_word < end_word:
                            start_word += 1
                else:   
                    # track the positions of each audio code in the larger codes_str sequence
                    audio_idx, _ = self._get_audio_only_codes_str(codes_str)
                    # yield examples from the sequence with the specified sequence length and overlap
                    start_code = 0
                    while True:
                        end_code = start_code + context_codes
                        start = audio_idx[start_code] if start_code > 0 else 0
                        end = audio_idx[end_code] if end_code < len(audio_idx) else len(codes_str)
                        example = codes_str[start:end]
                        # build the header
                        header = self._build_common_header(interleave_order, speakers)
                        if interleave_order == InterleaveOrder.AGENT:
                            agent_channel = channel_map.get(agent_speaker, {"channel": 0})["channel"]
                            agent_channel_isolated = is_speaker_channel_isolated(channel_map, agent_speaker)
                            agent_voice = self._select_agent_voice(
                                channels_chars[agent_channel], 
                                trans_pos_bounds[0] + int(start_code / num_channels), 
                                trans_pos_bounds[0] + int(end_code / num_channels), 
                                transcript_lines, 
                                agent_speaker,
                                agent_channel_isolated,
                            )
                            if agent_voice is not None:
                                header += f"{self.header_agent_voice_token}{agent_voice}"
                        example = f"{header}{self.header_end_token}{example}"
                        yield example, metadata.copy()
                        metadata["example_index"] += 1
                        if end_code >= len(audio_idx):
                            break
                        start_code = end_code - overlap_codes