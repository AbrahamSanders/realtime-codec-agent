import pylangacq
import re
import argparse
import os
import json
from tqdm import tqdm
from pylangacq.objects import Utterance

def clean_line(line, remove_bracketed=False):
    # convert <comma>, <period>, <questionmark>, <exclamationpoint> to punctuation"
    line = re.sub("<comma>", ",", line)
    line = re.sub("<period>", ".", line)
    line = re.sub("<questionmark>", "?", line)
    line = re.sub("<exclamationpoint>", "!", line)
    # convert <sil>, <music>, <noise>, <other> to [sil], [music], [noise], [other]
    line = re.sub("<sil>", "[sil]", line)
    line = re.sub("<music>", "[music]", line)
    line = re.sub("<noise>", "[noise]", line)
    line = re.sub("<other>", "[other]", line)
    # convert 'hello [!]' to 'hello!'
    line = re.sub(r" \[!\]", "!", line)
    if remove_bracketed:
        # get rid of bracketed sequences that don't contain a comment or sound
        line = re.sub(r"\[[^%\]].*?\]", "", line)
    # get rid of timestamp TODO: extract pauses from timestamp differences.
    line = re.sub(r"\d+?_\d+?", "", line)
    # get rid of +" and +,
    line = re.sub(r'\+[",]', "", line)
    # get rid of +/.
    line = re.sub(r'\+/\.', "", line)
    # get rid of &- (comes before fillers, e.g., 'um')
    line = re.sub("&-", "", line)
    # replace ° or ☺ or ⁎ with a single space
    line = re.sub("[°☺⁎]", " ", line)
    # get rid of any non word or non punctuation characters
    line = re.sub(r"[^\w !?.,;\"'`()&=%\-\[\]]", "", line)
    # get rid of ʔ which is somehow a word character 
    line = re.sub("ʔ", "", line)
    # get rid of "Long Events" notation and other specialized &= notations
    line = re.sub(r"&[l,n]=.+?(?=(?:\s|\Z))", "", line)
    line = re.sub(r"&=(?:lengthened|tsk|in|nonvocal|ex)(?=(?:\s|\Z))", "", line, flags=re.IGNORECASE)
    # get rid of ((...)) notation indicating that the annotator was not sure about the transcription
    line = re.sub(r"\(\( *(.*?) *\)\)", r"\1", line)
    # fix underscores following periods or used as periods in acronyms, e.g. "u._s._a." -> "u.s.a." or "u_s_a" -> "u.s.a"
    line = re.sub(r"(?<=[ _]\w)\.?_", ".", line)
    # special case of the above for the beginning of the string, because we can't put a fucking anchor in a character class 
    # and re can't use conditionals in a lookbehind 
    line = re.sub(r"(?<=\A\w)\.?_", ".", line)
    # replace remaining underscores with spaces
    line = re.sub("_", " ", line)
    # normalize sequences of spaces to a single space
    line = re.sub(" {2,}", " ", line)
    # close punctuation and contractions that have an extra space
    # between the word and the punctuation or contraction
    line = re.sub(" (?=[!?.,;'])", "", line)
    line = re.sub(" (?=n')", "", line)
    # finally, strip the line
    line = line.strip()
    return line

def expand_talkbank_utterances(utterances):
    expanded_utterances = []
    for utt in utterances:
        text = utt.tiers[utt.participant]
        text_time_marks = list(re.finditer(r"(\d+?)_(\d+?)", text))
        text_utts = [
            Utterance(
                participant=utt.participant,
                tokens=[], # we don't use this
                time_marks=(int(m.group(1)), int(m.group(2))),
                tiers={utt.participant: text[(text_time_marks[i-1].end() if i > 0 else 0):m.end()].lstrip()},
            )
            for i, m in enumerate(text_time_marks)
        ]
        if len(text_utts) > 0:
            # sanity check
            if text_utts[0].time_marks != utt.time_marks:
                raise ValueError(
                    f"Time marks of the first parsed utterance {text_utts[0].time_marks} do not match the original utterance time marks {utt.time_marks}."
                )
        if len(text_utts) > 1:
            expanded_utterances.extend(text_utts)
        else:
            # if there is only one utterance, just keep the original
            expanded_utterances.append(utt)
    return expanded_utterances

def get_talkbank_cleaned_utterances(header, utterances):
    participants = header["Participants"]
    part_map = {}
    for i, item in enumerate(participants.items()):
        part, _ = item
        part_map[part] = chr(ord("A") + i % 26)

    cleaned_utts = []
    utt_buffer = []
    for utt in utterances:
        # clean the utterance and prepend the speaker
        clean_utt = clean_line(utt.tiers[utt.participant], remove_bracketed=True)
        # some corpora have blank utterances in the format e.g., S1: 0. Skip these.
        if clean_utt in [".", "0."]:
            continue
        utt_buffer.append(f"{part_map[utt.participant]}: {clean_utt}")
        # if time_marks is None, wait and combine it with the next utterance that has time annotation
        if utt.time_marks is None:
            continue
        time_start, time_end = utt.time_marks
        # convert milliseconds to seconds
        time_start = time_start / 1000
        time_end = time_end / 1000
        cleaned_utts.extend([f"{time_start:.2f} {time_end:.2f} {utt}" for utt in utt_buffer])
        utt_buffer.clear()
    return cleaned_utts

def get_fisher_cleaned_utterances(transcript_file):
    cleaned_utts = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line_split = line.split()
            prefix  = " ".join(line_split[:3])
            text = " ".join(line_split[3:])
            text = clean_line(text, remove_bracketed=False)
            if not text:
                continue
            cleaned_utts.append(f"{prefix} {text}")
    return cleaned_utts

def lookup_fisher_partition(fisher_tran_part, number_subfolder):
    # What a fucking disaster.
    num = int(number_subfolder)
    if fisher_tran_part == "fe_03_p1_tran":
        if 0 <= num <= 7:
            return "fisher_eng_tr_sp_d1"
        if 8 <= num <= 16:
            return "fisher_eng_tr_sp_d2"
        if 17 <= num <= 25:
            return "fisher_eng_tr_sp_d3"
        if 26 <= num <= 34:
            return "fisher_eng_tr_sp_d4"
        if 35 <= num <= 43:
            return "fisher_eng_tr_sp_d5"
        if 44 <= num <= 52:
            return "fisher_eng_tr_sp_d6"
        if 53 <= num <= 58:
            return "fisher_eng_tr_sp_d7"
    if fisher_tran_part == "fe_03_p2_tran":
        if 58 <= num <= 66:
            return "fe_03_p2_sph1"
        if 67 <= num <= 75:
            return "fe_03_p2_sph2"
        if 76 <= num <= 83:
            return "fe_03_p2_sph3"
        if 84 <= num <= 91:
            return "fe_03_p2_sph4"
        if 92 <= num <= 99:
            return "fe_03_p2_sph5"
        if 100 <= num <= 108:
            return "fe_03_p2_sph6"
        if 109 <= num <= 116:
            return "fe_03_p2_sph7"
    raise ValueError(f"Unknown partition for {fisher_tran_part} and {number_subfolder}")

def get_gigaspeech_cleaned_utterances(segments):
    cleaned_utts = []
    speakers = [seg["speaker"] for seg in segments]
    part_map = {}
    for speaker in speakers:
        if speaker not in part_map:
            part_map[speaker] = chr(ord("A") + len(part_map) % 26)
    
    for seg in segments:
        text = seg["text_tn"].lower()
        clean_utt = clean_line(text, remove_bracketed=False)
        if not clean_utt:
            continue
        speaker = seg["speaker"]
        time_start = seg["begin_time"]
        time_end = seg["end_time"]
        cleaned_utts.append(f"{time_start:.2f} {time_end:.2f} {part_map[speaker]}: {clean_utt}")
    return cleaned_utts

def get_libriheavy_cleaned_utterances(supervisions, trans_start):
    cleaned_utts = []
    speakers = [sup["speaker"] for sup in supervisions]
    part_map = {}
    for speaker in speakers:
        if speaker not in part_map:
            part_map[speaker] = chr(ord("A") + len(part_map) % 26)

    for sup in supervisions:
        text = sup["custom"]["texts"][0]
        clean_utt = clean_line(text, remove_bracketed=False)
        if not clean_utt:
            continue
        speaker = sup["speaker"]
        time_start = trans_start + sup["start"]
        time_end = time_start + sup["duration"]
        cleaned_utts.append(f"{time_start:.2f} {time_end:.2f} {part_map[speaker]}: {clean_utt}")
    return cleaned_utts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare transcripts for creating LM dataset"
    )
    parser.add_argument("--transcripts_path", type=str, required=True)
    parser.add_argument("--sources", nargs="+", default=["talkbank", "fisher", "gigaspeech", "libriheavy"])
    args = parser.parse_args()

    raw_transcripts_path = os.path.join(args.transcripts_path, "raw")
    processed_transcripts_path = os.path.join(args.transcripts_path, "processed")
    os.makedirs(processed_transcripts_path, exist_ok=True)

    # First process all the TalkBank transcripts in .zip files
    if "talkbank" in args.sources:
        for file in os.listdir(raw_transcripts_path):
            if not file.endswith(".zip"):
                continue
            zip_file_path = os.path.join(raw_transcripts_path, file)
            print(f"Processing {zip_file_path}")

            target_folder = os.path.join(processed_transcripts_path, os.path.splitext(file)[0])
            os.makedirs(target_folder, exist_ok=True)
            
            reader = pylangacq.read_chat(zip_file_path)

            all_filepaths = reader.file_paths()
            all_headers = reader.headers()
            all_utterances = reader.utterances(by_files=True)
            for filepath, header, utterances in tqdm(zip(all_filepaths, all_headers, all_utterances), desc="Files"):
                utterances = expand_talkbank_utterances(utterances)
                cleaned_utts = get_talkbank_cleaned_utterances(header, utterances)
                # Save text file
                filename = os.path.splitext(os.path.basename(filepath))[0]
                out_filepath = os.path.join(target_folder, f"{filename}.txt")
                with open(out_filepath, "w", encoding="utf-8") as f:
                    for line in cleaned_utts:
                        f.write(line)
                        f.write("\n")

    # Next process all the Fisher transcripts
    if "fisher" in args.sources:
        for fisher_tran_part, fisher_audio_part in [("fe_03_p1_tran", "fisher_eng_tr_sp_LDC2004S13"), ("fe_03_p2_tran", "fe_03_p2_LDC2005S13")]:
            print(f"Processing {fisher_tran_part}")
            fisher_transcripts_path = os.path.join(raw_transcripts_path, fisher_tran_part, "data", "trans")
            for root, _, files in os.walk(fisher_transcripts_path):
                files = sorted([os.path.join(root, f) for f in files if os.path.splitext(f)[1] == ".txt"])
                if len(files) == 0:
                    continue
                for transcript_file in tqdm(files, desc=f"Files in {root}"):
                    cleaned_utts = get_fisher_cleaned_utterances(transcript_file)
                    # Save text file
                    number_subfolder = os.path.basename(os.path.dirname(transcript_file))
                    partition = lookup_fisher_partition(fisher_tran_part, number_subfolder)
                    target_path = os.path.join(processed_transcripts_path, fisher_audio_part, partition, "audio")
                    out_filepath = transcript_file.replace(fisher_transcripts_path, target_path)
                    # create the directory if it doesn't exist
                    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
                    with open(out_filepath, "w", encoding="utf-8") as f:
                        for line in cleaned_utts:
                            f.write(line)
                            f.write("\n")

    # Next process all the Gigaspeech transcripts
    if "gigaspeech" in args.sources:
        with open(os.path.join(raw_transcripts_path, "GigaSpeech.json"), "r", encoding="utf-8") as f:
            gigaspeech_transcripts = json.load(f)
        for audio in tqdm(gigaspeech_transcripts["audios"], desc="GigaSpeech"):
            audio_path = audio["path"]
            if not re.search("/podcast/P0000/", audio_path) and not re.search("/youtube/P00[0-3][0-9]/", audio_path):
                continue
            segments = audio["segments"]
            cleaned_utts = get_gigaspeech_cleaned_utterances(segments)
            # Save text file
            out_filepath = os.path.join(processed_transcripts_path, "gigaspeech", audio_path.replace(".opus", ".txt"))
            # create the directory if it doesn't exist
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
            with open(out_filepath, "w", encoding="utf-8") as f:
                for line in cleaned_utts:
                    f.write(line)
                    f.write("\n")

    # Finally process all the libriheavy transcripts
    if "libriheavy" in args.sources:
        libriheavy_transcripts = []
        with open(os.path.join(raw_transcripts_path, "libriheavy_cuts_small.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                libriheavy_transcripts.append(json.loads(line))
        # sort transcripts by file and start time
        libriheavy_transcripts.sort(key=lambda x: (x["recording"]["id"], x["start"]))
        libriheavy_transcripts.append(None)
        out_file_lines = []
        last_audio_path = None
        for transcript in tqdm(libriheavy_transcripts, desc="LibriHeavy"):
            audio_path = "libri-light-" + transcript["recording"]["id"] if transcript is not None else None
            if audio_path != last_audio_path and last_audio_path is not None:
                # Save text file
                out_filepath = os.path.join(processed_transcripts_path, f"{last_audio_path}.txt")
                # create the directory if it doesn't exist
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
                with open(out_filepath, "w", encoding="utf-8") as f:
                    for line in out_file_lines:
                        f.write(line)
                        f.write("\n")
                out_file_lines.clear()
            last_audio_path = audio_path
            if transcript is None:
                continue
            # append to out_file_lines with deduplication
            cleaned_utts = get_libriheavy_cleaned_utterances(transcript["supervisions"], transcript["start"])
            for line in cleaned_utts:
                last_out_file_line = None
                if len(out_file_lines) > 0:
                    last_out_file_line = out_file_lines[-1]
                    last_out_file_line_split = last_out_file_line.split()
                    last_start_time = float(last_out_file_line_split[0])
                    last_end_time = float(last_out_file_line_split[1])
                    last_speaker = last_out_file_line_split[2]
                line_split = line.split()
                start_time = float(line_split[0])
                end_time = float(line_split[1])
                speaker = line_split[2]
                if last_out_file_line is not None and last_start_time == start_time and last_speaker == speaker and end_time > last_end_time:
                    out_file_lines[-1] = line
                elif last_out_file_line is not None and last_end_time == end_time and last_speaker == speaker and start_time > last_start_time:
                    # skip this line
                    pass
                elif last_out_file_line is not None and start_time < last_end_time:
                    # skip this line
                    pass
                else:
                    out_file_lines.append(line)
