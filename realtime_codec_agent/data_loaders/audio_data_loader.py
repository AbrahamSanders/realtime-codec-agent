import os
import librosa
import torch
from transformers import EncodecModel, AutoProcessor, AutoTokenizer, AutoConfig
from tqdm import tqdm
from directory_downloader import DDownloader

from ..utils.encodec_utils import n_codebooks_to_bandwidth_id
from ..utils.tokenizer_utils import add_special_audio_tokens

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

class AudioDataLoader:
    def __init__(self, encodec_modelname, tokenizer_name, use_n_codebooks=2, tokenizer_offset=-1, add_audio_tokens=False, 
                 history_secs=20, overlap_secs=5, drop_last=True, download_dir=None, force_download=False):
        
        # Sanity checks
        if add_audio_tokens and tokenizer_offset != -1:
            raise ValueError("Cannot add audio tokens before the end of the tokenizer. Please set tokenizer_offset to -1.")

        # Encodec model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodec_model = EncodecModel.from_pretrained(encodec_modelname).to(self.device)
        self.encodec_processor = AutoProcessor.from_pretrained(encodec_modelname)
        self.use_n_codebooks = use_n_codebooks
        bandwidth_id = n_codebooks_to_bandwidth_id(self.use_n_codebooks)
        self.encodec_bandwidth = self.encodec_model.config.target_bandwidths[bandwidth_id]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_offset = tokenizer_offset
        if add_audio_tokens:
            config = AutoConfig.from_pretrained(tokenizer_name)
            added_tokens = add_special_audio_tokens(self.tokenizer, config.vocab_size, self.use_n_codebooks, self.encodec_model.config.codebook_size)
            print (f"Added {added_tokens} audio tokens to the tokenizer. New tokenizer size: {len(self.tokenizer)}")
        if self.tokenizer_offset == -1:
            self.tokenizer_offset = len(self.tokenizer) - self.use_n_codebooks * self.encodec_model.config.codebook_size
            print(f"Setting tokenizer_offset to {self.tokenizer_offset} (len(tokenizer) - use_n_codebooks * codebook_size).")
        
        # Data settings
        self.history_secs = history_secs
        self.overlap_secs = overlap_secs
        self.drop_last = drop_last
        self.corpora_urls = {
            "CallFriend_eng_n": "https://media.talkbank.org/ca/CallFriend/eng-n/",
            "CallFriend_eng_s": "https://media.talkbank.org/ca/CallFriend/eng-s/",
            "CallHome_eng": "https://media.talkbank.org/ca/CallHome/eng/",
            "SBCSAE": "https://media.talkbank.org/ca/SBCSAE/",
            "fisher_eng_tr_sp_LDC2004S13": None, # Fisher English Training Speech Part 1
            "fe_03_p2_LDC2005S13": None # Fisher English Training Speech Part 2
        }
        if download_dir is None:
            download_dir = "data/audio/raw"
        self.download_dir = download_dir
        self.force_download = force_download

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
                if corpus_url is None:
                    raise ValueError(f"Corpus '{corpus}' is not available for download.")
                downloader = DDownloader()
                await crawl(downloader, corpus_url, extensions=[".mp3"])
                await downloader.download_files(full_directory=corpus_path)

            audio_files = []
            for root, _, files in os.walk(corpus_path):
                audio_files.extend([os.path.join(root, file) for file in files if file.endswith(".mp3")])
            audio_files.sort()
            for audio_file in tqdm(audio_files, desc="Files"):
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
                    encoder_outputs = self.encodec_model.encode(**inputs, bandwidth=self.encodec_bandwidth).audio_codes # 1 x 1 x n_codebooks x n_tokens

                    audio_codes = torch.zeros(encoder_outputs.shape[-1] * self.use_n_codebooks, dtype=encoder_outputs.dtype).to(self.device)
                    for i in range(self.use_n_codebooks):
                        codebook_offset = i * self.encodec_model.config.codebook_size
                        audio_codes[i::self.use_n_codebooks] = self.tokenizer_offset + codebook_offset + encoder_outputs[0, 0, i]

                    example = self.tokenizer.decode(audio_codes, skip_special_tokens=False)
                    if group_by_dialogue:
                        dialogue.append(example)
                    else:
                        yield example
                    if end >= audio.shape[-1]:
                        break
                    start = end - self.overlap_secs * sr
                if group_by_dialogue and len(dialogue) > 0:
                    yield audio_file, dialogue
