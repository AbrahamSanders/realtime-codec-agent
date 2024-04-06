import os
import re
import librosa
import random
from tqdm import tqdm

from .audio_data_loader import AudioDataLoader

class AudioTextAlignDataLoader(AudioDataLoader):
    def __init__(self, *args, min_audio_context_secs=0.2, min_keep_target_words=3, transcripts_dir=None, random_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpora_transcripts = {
            "fisher_eng_tr_sp_LDC2004S13": "fe_03_p1_tran",
            "fe_03_p2_LDC2005S13": "fe_03_p2_tran"
        }
        if transcripts_dir is None:
            transcripts_dir = "data/transcripts/raw"
        self.min_audio_context_secs = min_audio_context_secs
        self.min_keep_target_words = min_keep_target_words
        self.transcripts_dir = transcripts_dir
        self.random_seed = random_seed

    async def load_data(self, corpora="All", group_by_dialogue=False):
        random.seed(self.random_seed)
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
                dialogue = self.create_audio_examples(audio, sr)
                dialogue_align = self.create_audio_align_examples(audio, sr, transcript_lines)
                dialogue.extend(dialogue_align)
                if not group_by_dialogue:
                    for example in dialogue:
                        yield example
                elif len(dialogue) > 0:
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
    
    def create_audio_align_examples(self, audio, sr, transcript_lines):
        examples = []
        avg_secs_per_word = self.get_average_secs_per_word(transcript_lines)
        last_trans_end_secs = 0
        for trans_start_secs, trans_end_secs, speaker, text in transcript_lines:
            if trans_start_secs < self.min_audio_context_secs:
                continue
            # we want an even distribution of audio context lengths, but we always need enough context to reasonably
            # predict the next words. So, we'll randomly choose a number of words to use from the transcript
            # to allow for shorter or longer audio contexts.
            trans_words = text.split()
            if len(trans_words) > self.min_keep_target_words:
                use_num_words = random.randint(self.min_keep_target_words, len(trans_words))
            else:
                use_num_words = len(trans_words)
            use_approx_length = use_num_words * avg_secs_per_word
            use_text = " ".join(trans_words[:use_num_words])
            # the longest audio context should stretch back until self.history_secs before the current transcript starts
            # or until the beginning of the audio file if within self.history_secs of the start
            min_audio_start_secs = max(0, trans_start_secs-self.history_secs)
            # the shortest audio context should include at least the same approximate audio length as the number of words
            # used in the current transcript, starting from the end of the previous transcript or the start of this one,
            # whichever comes first.
            max_audio_start_secs = min(last_trans_end_secs, trans_start_secs) - use_approx_length
            if max_audio_start_secs < min_audio_start_secs:
                max_audio_start_secs = min_audio_start_secs
            # randomly pick a start point within the allowed range to get a varied distribution of audio context lengths
            # for each prediction target length
            audio_start_secs = random.uniform(min_audio_start_secs, max_audio_start_secs)
            # cut the audio and make the example
            start = round(audio_start_secs * sr)
            end = round(trans_start_secs * sr)
            audio_slice = audio[..., start:end]
            example = self.tokenize_audio(audio_slice, sr)
            example += f" {speaker}: {use_text}"
            examples.append(example)
            last_trans_end_secs = trans_end_secs
        return examples
    
    def get_average_secs_per_word(self, transcript_lines):
        total_secs = 0
        total_words = 0
        for trans_start_secs, trans_end_secs, _, text in transcript_lines:
            total_secs += trans_end_secs - trans_start_secs
            total_words += len(text.split())
        return total_secs / total_words