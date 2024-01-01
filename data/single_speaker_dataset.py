"""Main loading function.

Copyright PolyAI Limited.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from librosa.util import normalize
from pyannote.audio import Inference
from torch.utils import data

import constants as c


def random_crop(x, maxseqlen):
    if x.shape[0] >= maxseqlen:
        offset = random.randrange(x.shape[0] - maxseqlen + 1)
        x = x[offset: offset + maxseqlen]
    else:
        offset = 0
    return x, offset


def dynamic_range_compression(x, C=0.3, M=6.5, clip_val=1e-5):
    return (np.log(np.clip(x, a_min=clip_val, a_max=None)) + M) * C


def dynamic_range_decompression(x, C=0.3, M=6.5):
    return np.exp(x / C - M)


class QuantizeDataset(data.Dataset):
    def __init__(self, hp, metapath, datadir=None, speaker_embedding_dir=None):
        self.hp = hp
        self.datadir = Path(datadir)
        self.speaker_embedding_dir = speaker_embedding_dir
        self.sem_mask_id = hp.n_semantic_codes

        print(f"Loading metadata in {metapath}...")
        with open(metapath, "r") as f:
            self.text = json.load(f)
        if 0 < self.hp.max_dataset_samples < len(self.text):
            self.new_text = {}
            num = 0
            for k, v in self.text.items():
                if num >= self.hp.max_dataset_samples:
                    break
                self.new_text[k] = v
                num += 1
            self.text = self.new_text

        self.datasetbase = [x for x in self.text.keys()]
        self.dataset = [
            os.path.join(self.datadir, x) for x in self.datasetbase]

        if self.speaker_embedding_dir is None:
            self.spkr_embedding = Inference(
                "pyannote/embedding",
                window="whole",
                use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            )

        # Print statistics:
        n = len(self.dataset)
        print(f"Total {n} examples")

        self.lengths = [float(v["duration"]) for v in self.text.values()]
        total_duration = sum(self.lengths)
        avglen = total_duration / len(self.lengths)
        maxlen = max(self.lengths)
        minlen = min(self.lengths)
        print(
            f"Average duration of audio: {avglen} sec, "
             "Maximum duration: {maxlen} sec, Minimum duration: {minlen} sec"
        )

    def __len__(self):
        return len(self.dataset)

    def load_quantization(self, _name):
        if self.hp.vocoder_type == 'NATIVE':
            metadata = self.text[_name]
            quantization = np.array(metadata["quantization"]).T  # ..., 4
        elif self.hp.vocoder_type == 'DAC':
            codes_path = self.datadir.parent / 'audios-dac' / (os.path.splitext(_name)[0] + ".npy")  # noqa
            quantization = np.load(codes_path).T  # ..., 12
        elif self.hp.vocoder_type == 'ENCODEC':
            codes_path = self.datadir.parent / 'audios-encodec' / (os.path.splitext(_name)[0] + ".npy")  # noqa
            quantization = np.load(codes_path).squeeze(0).T  # ..., 8
        elif self.hp.vocoder_type == 'SPEECHTOKENIZER':
            codes_path = self.datadir.parent / 'audios-speech-tokenizer/acoustic' / (os.path.splitext(_name)[0] + ".npy")  # noqa
            quantization = np.load(codes_path).T  # ..., 7
        else:
            raise ValueError(f"Unknown vocoder_type {self.hp.vocoder_type}")

        return quantization

    def __getitem__(self, i):
        dataname = self.dataset[i]
        _name = self.datasetbase[i]
        metadata = self.text[_name]

        # Speaker 1 
        acoustic_tokens = self.load_quantization(_name)
        acoustic_tokens = np.pad(
            acoustic_tokens, [[1, 0],[0,0]], constant_values=c.SPKR_1)

        npy_path = self.datadir.parent / 'audios-speech-tokenizer/semantic' / (os.path.splitext(_name)[0] + ".npy")    # noqa
        semantic_tokens = np.load(npy_path)[None]
        semantic_tokens = np.pad(
            semantic_tokens,[[0,0], [1, 0]], constant_values=c.SPKR_1)

        if "name_2" in metadata:
            wav, _ = sf.read(dataname.split(".")[0] + "_1.wav")
        else:
            wav, _ = sf.read(dataname)
        audio = normalize(wav) * 0.95
        speaker_embedding = self.spkr_embedding(
            {"waveform": torch.FloatTensor(audio).unsqueeze(0),
             "sample_rate": self.hp.sample_rate,}
        ).reshape(1, -1)
        speaker_embedding = np.repeat(
            speaker_embedding, semantic_tokens.shape[1], axis=0)

        # Speaker 2 
        if "text_2" in metadata:
            _name = _name.split(".wav")[0] + "_2.wav"
            acoustic_tokens_2 = self.load_quantization(_name)
            acoustic_tokens_2 = np.pad(
                acoustic_tokens_2, [[1, 0],[0,0]], constant_values=c.SPKR_2)

            npy_path = self.datadir.parent / 'audios-speech-tokenizer/semantic' / (os.path.splitext(_name)[0] + ".npy")  # noqa
            semantic_tokens_2 = np.load(npy_path)[None]
            semantic_tokens_2 = np.pad(
                semantic_tokens_2,[[0,0], [1, 0]], constant_values=c.SPKR_2)
            
            wav, _ = sf.read(dataname.split(".wav")[0] + "_2.wav")
            audio = normalize(wav) * 0.95
            speaker_embedding_2 = self.spkr_embedding(
                {"waveform": torch.FloatTensor(audio).unsqueeze(0), 
                 "sample_rate": self.hp.sample_rate,}
            ).reshape(1, -1)
            speaker_embedding_2 = np.repeat(
                speaker_embedding_2, semantic_tokens_2.shape[1], axis=0)

            # Merge both speakers
            acoustic_tokens = np.concatenate(
                (acoustic_tokens, acoustic_tokens_2), axis=0)
            semantic_tokens = np.concatenate(
                (semantic_tokens, semantic_tokens_2), axis=1)
            speaker_embedding = np.concatenate(
                (speaker_embedding, speaker_embedding_2), axis=0)

        speaker_embedding = speaker_embedding[:self.hp.max_length, :]
        acoustic_tokens = acoustic_tokens[:self.hp.max_length, :]
        semantic_tokens = semantic_tokens[:, :self.hp.max_length]

        # # HACK - we have no 8 lvls pfb30
        # acoustic_tokens = np.concatenate((semantic_tokens.T, acoustic_tokens), axis=1)
        # # END HACK
    
        return speaker_embedding, acoustic_tokens, acoustic_tokens, dataname, semantic_tokens  # noqa
