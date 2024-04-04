import os
import re
import librosa
from tqdm import tqdm

from .audio_data_loader import AudioDataLoader

class AudioTextAlignDataLoader(AudioDataLoader):
    def __init__(self, *args, min_audio_context_secs=0.2, transcripts_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpora_transcripts = {
            "fisher_eng_tr_sp_LDC2004S13": "fe_03_p1_tran",
            "fe_03_p2_LDC2005S13": "fe_03_p2_tran"
        }
        if transcripts_dir is None:
            transcripts_dir = "data/transcripts/raw"
        self.min_audio_context_secs = min_audio_context_secs
        self.transcripts_dir = transcripts_dir

    async def load_data(self, corpora="All", group_by_dialogue=False):
        if isinstance(corpora, str):
            if corpora == "All":
                corpora = list(self.corpora_transcripts)
            else:
                corpora = corpora.split(",")
            
        for corpus in corpora:
            if corpus not in self.corpora_transcripts:
                raise ValueError(f"Corpus '{corpus}' is not currently supported. "
                                 f"Choose from {list(self.corpora_transcripts)}, passed as a list "
                                 "or a comma delimited string, or pass 'All'.")
        
        for corpus in tqdm(corpora, desc="Corpora"):
            corpus_path = os.path.join(self.download_dir, corpus)
            transcripts_path = os.path.join(self.transcripts_dir, self.corpora_transcripts[corpus])
            # TODO: For TalkBank, support downloading audio & transcripts together. For now we'll assume everything is downloaded.
            if not os.path.exists(corpus_path):
                raise ValueError(f"Corpus '{corpus}' audio files are not available. Please download them first.")
            if not os.path.exists(transcripts_path):
                raise ValueError(f"Corpus '{corpus}' transcripts are not available. Please download them first.")
            
            audio_files = self.get_audio_files(corpus_path)
            transcript_files = self.get_transcript_files(transcripts_path, audio_files)
            for audio_file, transcript_file in tqdm(zip(audio_files, transcript_files), desc="Files"):
                audio, sr = librosa.load(audio_file, sr=self.encodec_model.config.sampling_rate, mono=True)
                transcript_lines = self.load_transcript(transcript_file)
                
                if group_by_dialogue:
                    dialogue = []
                for trans_start_secs, _, speaker, text in transcript_lines:
                    if trans_start_secs < self.min_audio_context_secs:
                        continue
                    audio_start_secs = max(0, trans_start_secs - self.history_secs)
                    start = round(audio_start_secs * sr)
                    end = round(trans_start_secs * sr)
                    audio_slice = audio[..., start:end]
                    example = self.tokenize_audio(audio_slice, sr)
                    example += f" {speaker}: {text}"
                    if group_by_dialogue:
                        dialogue.append(example)
                    else:
                        yield example
                if group_by_dialogue and len(dialogue) > 0:
                    yield audio_file, dialogue

    def get_transcript_files(self, transcripts_path, audio_files):
        transcript_files = []
        for audio_file in audio_files:
            file = os.path.basename(audio_file)
            parent_dir = os.path.basename(os.path.dirname(audio_file))
            transcript_file = os.path.join(transcripts_path, "data", "trans", parent_dir, file.replace(".mp3", ".txt"))
            if not os.path.exists(transcript_file):
                raise ValueError(f"Missing transcript {transcript_file}.")
            transcript_files.append(transcript_file)
        return transcript_files
    
    def load_transcript(self, transcript_file):
        transcript_lines = []
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line_split = line.split()
                start_secs, end_secs, speaker = float(line_split[0]), float(line_split[1]), line_split[2].rstrip(":")
                text = " ".join(line_split[3:])
                # get rid of ((...)) notation indicating that the annotator was not sure about the transcription
                text = re.sub(r"\(\( *(.*?) *\)\)", r"\1", text)
                # normalize sequences of spaces to a single space
                text = re.sub(" {2,}", " ", text)
                text = text.strip()
                if not text:
                    continue
                transcript_lines.append((start_secs, end_secs, speaker, text))
        return transcript_lines