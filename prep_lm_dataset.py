import argparse
import os
import functools
import jsonlines
from tqdm import tqdm

from realtime_codec_agent.lm_dataset_builder import LMDatasetBuilder, InterleaveOrder
from codec_bpe import UNICODE_OFFSET
from codec_bpe.core.utils import get_codec_info, update_args_from_codec_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use numpy files containing audio codes to construct a plain-text codec BPE dataset suitable for language modeling"
    )
    parser.add_argument("--codes_path", type=str, required=True)
    parser.add_argument("--transcripts_path", type=str, required=True)
    parser.add_argument("--num_codebooks", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--codec_framerate", type=float, default=None)
    parser.add_argument("--interleave_order", type=str, choices=list(InterleaveOrder), default=InterleaveOrder.ALL)
    parser.add_argument("--audio_start_token", type=str, default="<|audio|>")
    parser.add_argument("--audio_end_token", type=str, default="<|end_audio|>")
    parser.add_argument("--header_audio_only_token", type=str, default="<|audio_only|>")
    parser.add_argument("--header_text_only_token", type=str, default="<|text_only|>")
    parser.add_argument("--header_audio_first_token", type=str, default="<|audio_first|>")
    parser.add_argument("--header_text_first_token", type=str, default="<|text_first|>")
    parser.add_argument("--header_agent_token", type=str, default="<|agent|>")
    parser.add_argument("--header_agent_voice_token", type=str, default="<|agent_voice|>")
    parser.add_argument("--header_speaker_token", type=str, default="<|speaker|>")
    parser.add_argument("--header_end_token", type=str, default="<|end_header|>")
    # handle hex values for unicode_offset with argparse: https://stackoverflow.com/a/25513044
    parser.add_argument("--unicode_offset", type=functools.partial(int, base=0), default=UNICODE_OFFSET)
    parser.add_argument("--context_secs", type=float, default=80.0)
    parser.add_argument("--overlap_secs", type=float, default=20.0)
    parser.add_argument("--text_only_context_words", type=int, default=3000)
    parser.add_argument("--text_only_overlap_words", type=int, default=750)
    parser.add_argument("--max_voice_enrollment_secs", type=float, default=6.0)
    parser.add_argument("--voice_enrollment_selection_seed", type=int, default=42)
    parser.add_argument("--agent_identity", type=str, default="A")
    parser.add_argument("--speaker_proportion_threshold", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="output/lm_dataset.txt")
    parser.add_argument("--codes_filter", type=str, nargs="+")
    parser.add_argument("--codes_filter_exclude", type=str, nargs="+")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    codec_info = get_codec_info(args.codes_path)
    update_args_from_codec_info(args, codec_info)
    if args.num_codebooks is None or args.codebook_size is None or args.codec_framerate is None:
        raise ValueError(
            "codec_info.json does not exist in --codes_path so you must specify --num_codebooks, --codebook_size, and --codec_framerate manually."
        )

    lm_dataset_builder = LMDatasetBuilder(
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        codec_framerate=args.codec_framerate,
        interleave_order=args.interleave_order,
        audio_start_token=args.audio_start_token,
        audio_end_token=args.audio_end_token,
        header_audio_only_token=args.header_audio_only_token,
        header_text_only_token=args.header_text_only_token,
        header_audio_first_token=args.header_audio_first_token,
        header_text_first_token=args.header_text_first_token,
        header_agent_token=args.header_agent_token,
        header_agent_voice_token=args.header_agent_voice_token,
        header_speaker_token=args.header_speaker_token,
        header_end_token=args.header_end_token,
        unicode_offset=args.unicode_offset,
        context_secs=args.context_secs,
        overlap_secs=args.overlap_secs,
        text_only_context_words=args.text_only_context_words,
        text_only_overlap_words=args.text_only_overlap_words,
        max_voice_enrollment_secs=args.max_voice_enrollment_secs,
        voice_enrollment_selection_seed=args.voice_enrollment_selection_seed,
        agent_identity=args.agent_identity,
        speaker_proportion_threshold=args.speaker_proportion_threshold,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    metadata_path = args.save_path.replace(".txt", "_metadata.jsonl")

    with open(args.save_path, "w", encoding="utf-8") as f:
        with jsonlines.open(metadata_path, "w") as f_meta:
            example_iterator = lm_dataset_builder.iterate_examples(
                args.codes_path, args.transcripts_path, args.codes_filter, args.codes_filter_exclude
            )
            for i, (example, metadata) in tqdm(enumerate(example_iterator), desc="Examples"):
                if i == args.num_examples:
                    break
                f.write(example)
                f.write("\n")

                f_meta.write(metadata)
            