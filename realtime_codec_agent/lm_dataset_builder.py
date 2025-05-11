from typing import Optional, Union, Iterator, List, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
from enum import Enum
import numpy as np
import os
import itertools

from codec_bpe.core.converter import codes_to_chars, UNICODE_OFFSET
from codec_bpe.core.utils import get_codes_files

class InterleaveOrder(Enum):
    AUDIO_FIRST = "audio_first"
    TEXT_FIRST = "text_first"
    BOTH = "both"

class LMDatasetBuilder:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: float,
        interleave_order: InterleaveOrder = InterleaveOrder.BOTH,
        audio_start_token: Optional[str] = "<|audio|>",
        audio_end_token: Optional[str] = "<|end_audio|>",
        header_audio_first_token: Optional[str] = "<|audio_first|>",
        header_text_first_token: Optional[str] = "<|text_first|>",
        header_speaker_token: Optional[str] = "<|speaker|>",
        header_end_token: Optional[str] = "<|end_header|>",
        unicode_offset: int = UNICODE_OFFSET,
        sequence_length: int = 4096,
        overlap_length: int = 1024,
        drop_last: bool = False,
        verify_sequence_length: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.interleave_order = interleave_order
        self.unicode_offset = unicode_offset
        self.sequence_length = sequence_length
        self.overlap_length = overlap_length
        self.drop_last = drop_last
        self.verify_sequence_length = verify_sequence_length

        self.audio_start_token = audio_start_token
        if audio_start_token is not None and self.tokenizer.convert_tokens_to_ids(audio_start_token) is None:
            raise ValueError(f"Token '{audio_start_token}' not found in tokenizer")
        self.audio_end_token = audio_end_token
        if audio_end_token is not None and self.tokenizer.convert_tokens_to_ids(audio_end_token) is None:
            raise ValueError(f"Token '{audio_end_token}' not found in tokenizer")
        self.header_audio_first_token = header_audio_first_token
        if header_audio_first_token is not None and self.tokenizer.convert_tokens_to_ids(header_audio_first_token) is None:
            raise ValueError(f"Token '{header_audio_first_token}' not found in tokenizer")
        self.header_text_first_token = header_text_first_token
        if header_text_first_token is not None and self.tokenizer.convert_tokens_to_ids(header_text_first_token) is None:
            raise ValueError(f"Token '{header_text_first_token}' not found in tokenizer")
        self.header_speaker_token = header_speaker_token
        if header_speaker_token is not None and self.tokenizer.convert_tokens_to_ids(header_speaker_token) is None:
            raise ValueError(f"Token '{header_speaker_token}' not found in tokenizer")
        self.header_end_token = header_end_token
        if header_end_token is not None and self.tokenizer.convert_tokens_to_ids(header_end_token) is None:
            raise ValueError(f"Token '{header_end_token}' not found in tokenizer")

    def _group_codes_files(self, codes_files: List[str]) -> List[Tuple[str, List[List[str]]]]:
        grouped_codes_files = []
        last_file_root = None
        for codes_file in codes_files:
            codes_file_parts = codes_file.split("_")
            file_root = "_".join(codes_file_parts[:-2])
            channel = int(codes_file_parts[-2].lstrip("c"))
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
    
    def _load_transcript(self, transcript_file: str) -> List[Tuple[float, float, str, str]]:
        transcript_lines = []
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line_split = line.split()
                    start_secs, end_secs, speaker = float(line_split[0]), float(line_split[1]), line_split[2].rstrip(":")
                    text = " ".join(line_split[3:])
                    text = text.strip()
                    if not text:
                        continue
                    transcript_lines.append((start_secs, end_secs, speaker, text))
        # make sure the lines are in the correct order, first by start time, then by end time
        transcript_lines.sort(key=lambda x: (x[0], x[1]))
        return transcript_lines

    def _build_codes_strs(
        self, 
        codes: np.ndarray, 
        transcript_lines: List[Tuple[float, float, str, str]], 
        file_root: str,
    ) -> List[Tuple[str, InterleaveOrder]]:
        if not transcript_lines:
            print(f"No transcript found for {file_root}. Skipping file...")
            return []
        
        # convert codes to unicode string
        channels_chars = [
            codes_to_chars(
                ch_codes, 
                self.codebook_size, 
                copy_before_conversion=False,
                unicode_offset=self.unicode_offset,
            ) for ch_codes in codes
        ]
        
        # add a dummy line to handle any audio beyond the last transcribed line
        transcript_lines.append((None, None, None, None))

        # build the codes strings
        codes_strs = []
        if self.interleave_order == InterleaveOrder.AUDIO_FIRST or self.interleave_order == InterleaveOrder.BOTH:
            # build the audio-first codes string
            codes_str = self._build_codes_str(channels_chars, transcript_lines, InterleaveOrder.AUDIO_FIRST)
            codes_strs.append((codes_str, InterleaveOrder.AUDIO_FIRST))
        if self.interleave_order == InterleaveOrder.TEXT_FIRST or self.interleave_order == InterleaveOrder.BOTH:
            # build the text-first codes string
            codes_str = self._build_codes_str(channels_chars, transcript_lines, InterleaveOrder.TEXT_FIRST)
            codes_strs.append((codes_str, InterleaveOrder.TEXT_FIRST))
        return codes_strs

    def _build_codes_str(
        self, 
        channels_chars: List[str], 
        transcript_lines: List[Tuple[float, float, str, str]], 
        interleave_order: InterleaveOrder,
    ) -> str:
        if interleave_order == InterleaveOrder.BOTH:
            raise ValueError("InterleaveOrder.BOTH cannot be passed here, pass a specific interleave order")
        # build the codes string
        str_parts = []
        last_codes_pos = 0
        for start_secs, end_secs, speaker, text in transcript_lines:
            transcript_start_secs = start_secs if interleave_order == InterleaveOrder.TEXT_FIRST else end_secs
            if transcript_start_secs is None:
                line_start_pos = len(channels_chars[0])
            else:
                # Add any audio up to the point that the transcript line starts
                line_start_pos = int(transcript_start_secs * self.codec_framerate * self.num_codebooks)
                # Roll back to the last complete acoustic unit if we land in the middle of one
                line_start_pos -= line_start_pos % self.num_codebooks
            if line_start_pos > last_codes_pos:
                # add audio start token if specified
                if self.audio_start_token is not None:
                    str_parts.append(self.audio_start_token)
                audio_part = [chars[last_codes_pos:line_start_pos] for chars in channels_chars]
                audio_part = list(itertools.chain.from_iterable(zip(*audio_part)))
                audio_part = "".join(audio_part)
                str_parts.append(audio_part)
                last_codes_pos = line_start_pos
                # add audio end token if specified
                if self.audio_end_token is not None:
                    str_parts.append(self.audio_end_token)
            if speaker is not None:
                # Add the transcript line
                str_parts.append(f" {speaker}: {text}")

        # compile the codes string
        codes_str = "".join(str_parts)
        return codes_str

    def _is_audio_token(self, token: str) -> bool:
        return token == self.audio_start_token \
            or token == self.audio_end_token \
            or ord(token[0]) >= self.unicode_offset

    def iterate_examples(self, codes_path: str, transcripts_path: str, codes_filter: Optional[Union[str, List[str]]] = None) -> Iterator[str]:
        codes_files = get_codes_files(codes_path, codes_filter)
        # group codes files by root filename (minus channel and starting timestamp) and then by channel
        grouped_codes_files = self._group_codes_files(codes_files)
        for file_root, file_channels in tqdm(grouped_codes_files, desc="Codes file groups"):
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

            # load the transcript if it exists
            transcript_file = file_root.replace(codes_path, transcripts_path) + ".txt"
            transcript_lines = self._load_transcript(transcript_file)
            speakers = sorted(set([line[2] for line in transcript_lines]))

            # build the codes strings (unicode characters + additional tokens)
            codes_strs = self._build_codes_strs(codes, transcript_lines, file_root)
            for codes_str, interleave_order in codes_strs:
                # encode the codes string with the tokenizer
                tokens = self.tokenizer.encode(codes_str, return_tensors="np")[0]
                sequence_length = self.sequence_length
                if self.tokenizer.bos_token_id is not None and tokens[0] == self.tokenizer.bos_token_id:
                    tokens = tokens[1:]
                    sequence_length -= 1
                if self.header_audio_first_token is not None or self.header_text_first_token is not None:
                    sequence_length -= 1
                if self.header_speaker_token is not None:
                    sequence_length -= 2*len(speakers)
                if self.header_end_token is not None:
                    sequence_length -= 1
                # yield examples from the sequence with the specified sequence length and overlap
                start = 0
                while True:
                    end = start + sequence_length
                    while True:
                        first_tokens = self.tokenizer.convert_ids_to_tokens(tokens[start:start+2])
                        # Example must start from the beginning of the sequence or at an audio token. We don't want to start the example
                        # in the middle of text because it could cause words to be split across examples. If the example starts with a partial
                        # word token, its sequence length can be different than expected when encoded later in the training script.
                        # e.g., george is tokenized as [_ge, orge]. If the previous example ends at "_ge" and this example starts at "orge", 
                        # it will be re-tokenized as [_or, ge] instead of [orge] increasing the sequence length by 1 and requiring the training
                        # script to use padding or packing strategies to handle the mismatch.
                        if start == 0 or self._is_audio_token(first_tokens[0]):
                            # If the example starts with the last audio token before text, move forward to the beginning of the text
                            if start > 0 and not self._is_audio_token(first_tokens[1]):
                                start += 1
                                end += 1
                            break
                        start -= 1
                        end -= 1
                        
                    if self.drop_last and end > len(tokens):
                        break
                    example = self.tokenizer.decode(tokens[start:end], clean_up_tokenization_spaces=False)
                    # Build the header
                    header = ""
                    if interleave_order == InterleaveOrder.AUDIO_FIRST and self.header_audio_first_token is not None:
                        header += self.header_audio_first_token
                    elif interleave_order == InterleaveOrder.TEXT_FIRST and self.header_text_first_token is not None:
                        header += self.header_text_first_token
                    if self.header_speaker_token is not None:
                        header += "".join([f"{self.header_speaker_token}{speaker}" for speaker in speakers])
                    if self.header_end_token is not None:
                        header += self.header_end_token
                    example = f"{header}{example}"
                    if self.drop_last and self.verify_sequence_length:
                        # Sanity check: the tokenized example must have length of self.sequence_length
                        tokenized_example = self.tokenizer.encode(example, return_tensors="np")
                        if tokenized_example.shape[-1] != self.sequence_length:
                            raise RuntimeError(
                                f"Tokenized example length {tokenized_example.shape[-1]} does not match expected length {self.sequence_length}\n"
                                f"Example: {example}"
                            )
                    yield example
                    if end >= len(tokens):
                        break
                    start = end - self.overlap_length