from typing import List, Tuple, Dict, Any
import json
import os

def load_transcript(
    transcript_file: str, 
    speaker_proportion_threshold: float = 0.0,
) -> Tuple[List[Tuple[float, float, str, str]], List[str], Dict[str, Any]]:
    transcript_lines = []
    speaker_durations = {}
    # read the transcript
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
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + (end_secs - start_secs)
    
    # read the channel map if it exists
    channel_map_file = transcript_file.replace(".txt", "_channel_map.json")
    channel_map = {}
    if os.path.exists(channel_map_file):
        with open(channel_map_file, "r", encoding="utf-8") as f:
            channel_map = json.load(f)
    
    # filter out speakers with too short durations
    total_duration_secs = sum(speaker_durations.values())
    for speaker, duration_secs in sorted(speaker_durations.items(), key=lambda x: x[1]):
        if duration_secs / total_duration_secs < speaker_proportion_threshold:
            del speaker_durations[speaker]
    # remap speaker identities to ensure there are no gaps caused by filtered out speakers 
    # (e.g. we want A, B, C instead of A, C, E)
    speaker_map = {speaker: chr(ord("A") + i % 26) for i, speaker in enumerate(sorted(speaker_durations))}
    transcript_lines = [
        (i, start_secs, end_secs, speaker_map[speaker], text) 
        for i, (start_secs, end_secs, speaker, text) in enumerate(transcript_lines)
        if speaker in speaker_map
    ]
    channel_map = {speaker_map[speaker]: channel for speaker, channel in channel_map.items() if speaker in speaker_map}
    speakers = sorted(speaker_map.values())
    # make sure the lines are in the correct order, first by start time, then by end time, then by index
    transcript_lines.sort(key=lambda x: (x[1], x[2], x[0]))
    transcript_lines = [line[1:] for line in transcript_lines]  # remove index from the tuple
    return transcript_lines, speakers, channel_map

def is_speaker_channel_isolated(channel_map: Dict[str, Any], speaker: str) -> bool:
    """
    Check if the given speaker's channel is isolated (i.e., no other speakers share the same channel).
    """
    if speaker not in channel_map:
        return False
    speaker_channel = channel_map[speaker]["channel"]
    for other_speaker in channel_map:
        if other_speaker != speaker and channel_map[other_speaker]["channel"] == speaker_channel:
            return False
    return True

def set_agent_speaker(transcript_lines, speakers, channel_map, agent_speaker):
    if agent_speaker == "A":
        # Nothing to do here, agent is already A
        return transcript_lines, channel_map
    elif agent_speaker not in speakers:
        raise ValueError(f"Agent speaker {agent_speaker} not found in transcript speakers: {speakers}")
    # Swap the agent speaker with A in the transcript lines and channel map
    swapped_transcript_lines = []
    for start_secs, end_secs, speaker, text in transcript_lines:
        if speaker == agent_speaker:
            swapped_transcript_lines.append((start_secs, end_secs, "A", text))
        elif speaker == "A":
            swapped_transcript_lines.append((start_secs, end_secs, agent_speaker, text))
        else:
            swapped_transcript_lines.append((start_secs, end_secs, speaker, text))
    swapped_channel_map = {}
    for speaker, channel in channel_map.items():
        if speaker == agent_speaker:
            swapped_channel_map["A"] = channel
        elif speaker == "A":
            swapped_channel_map[agent_speaker] = channel
        else:
            swapped_channel_map[speaker] = channel
    return swapped_transcript_lines, swapped_channel_map