import os
import librosa
import torch
from transformers import EncodecModel, AutoProcessor, AutoTokenizer
from tqdm import tqdm
from directory_downloader import DDownloader

# Fix directory_downloader crawl function to accept kwargs supported by get_page_links.
# The documentation indicates it should work this way but was never implemented in the package.
async def crawl(downloader, url: str, *args, **kwargs):

    """ crawl a website and search of downloadables files
        url:str -> the directory link
    """

    links = await downloader.get_page_links(url, *args, **kwargs)
    for link in links:
        if link not in downloader.crawled_links:
            downloader.crawled_links.add(link)
            await crawl(downloader, link, *args, **kwargs)

class TalkbankDataLoader:
    def __init__(self, tokenizer_name, tokenizer_offset=0, history_secs=20, overlap_secs=5, drop_last=True, download_dir=None, force_download=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_offset = tokenizer_offset
        
        self.history_secs = history_secs
        self.overlap_secs = overlap_secs
        self.drop_last = drop_last
        self.corpora_urls = {
            "CallFriend_eng_n": "https://media.talkbank.org/ca/CallFriend/eng-n/",
            "CallFriend_eng_s": "https://media.talkbank.org/ca/CallFriend/eng-s/",
            "CallHome_eng": "https://media.talkbank.org/ca/CallHome/eng/",
            "SBCSAE": "https://media.talkbank.org/ca/SBCSAE/"
        }
        if download_dir is None:
            download_dir = "data/audio/raw"
        self.download_dir = download_dir
        self.force_download = force_download

        # Encodec model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    async def load_data(self, corpora="All", group_by_dialogue=False):
        if isinstance(corpora, str):
            if corpora == "All":
                corpora = list(self.corpora_urls)
            else:
                corpora = corpora.split(",")
            
        for corpus in corpora:
            if corpus not in self.corpora_urls:
                raise ValueError(f"Corpus '{corpus}' is not currently supported. "
                                 f"Choose from {list(self.corpora_urls)}, passed as a list "
                                 "or a comma delimited string, or pass 'All'.")
        
        for corpus in tqdm(corpora, desc="Corpora"):
            corpus_path = os.path.join(self.download_dir, corpus)
            if not os.path.exists(corpus_path) or self.force_download:
                corpus_url = self.corpora_urls[corpus]
                downloader = DDownloader()
                await crawl(downloader, corpus_url, extensions=[".mp3"])
                await downloader.download_files(full_directory=corpus_path)

            for audio_file in tqdm(os.listdir(corpus_path), desc="Files"):
                audio_file = os.path.join(corpus_path, audio_file)
                audio, sr = librosa.load(audio_file, sr=self.encodec_model.config.sampling_rate, mono=True)

                if group_by_dialogue:
                    dialogue = []
                start = 0
                while True:
                    end = start + self.history_secs * sr
                    if self.drop_last and end > audio.shape[-1]:
                        break

                    audio_slice = audio[..., start:end]
                    inputs = self.encodec_processor(raw_audio=audio_slice, sampling_rate=sr, return_tensors="pt").to(self.device)
                    encoder_outputs = self.encodec_model.encode(**inputs, bandwidth=1.5).audio_codes # 1 x 1 x n_codebooks x n_tokens

                    n_codebooks = encoder_outputs.shape[-2]
                    audio_codes = torch.zeros(encoder_outputs.shape[-1] * n_codebooks, dtype=encoder_outputs.dtype).to(self.device)
                    for i in range(n_codebooks):
                        codebook_offset = i * self.encodec_model.config.codebook_size
                        audio_codes[i::n_codebooks] = self.tokenizer_offset + codebook_offset + encoder_outputs[0, 0, i]

                    example = self.tokenizer.decode(audio_codes, skip_special_tokens=False)
                    if group_by_dialogue:
                        dialogue.append(example)
                    else:
                        yield example
                    if end >= audio.shape[-1]:
                        break
                    start = end - self.overlap_secs * sr
                if group_by_dialogue and len(dialogue) > 0:
                    yield dialogue
