"""Vocoder wrapper.

Copyright PolyAI Limited.
"""
import enum

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from speechtokenizer import SpeechTokenizer


class VocoderType(enum.Enum):
    SPEECHTOKENIZER = ("SPEECHTOKENIZER", 320)

    def __init__(self, name, compression_ratio):
        self._name_ = name
        self.compression_ratio = compression_ratio

    def get_vocoder(self, ckpt_path, config_path, **kwargs):
        if self.name == "SPEECHTOKENIZER":
            if ckpt_path:
                vocoder = STWrapper(ckpt_path, config_path)
            else:
                vocoder = STWrapper()
        else:
            raise ValueError(f"Unknown vocoder type {self.name}")
        return vocoder


class STWrapper(nn.Module):
    def __init__(
            self, 
            ckpt_path: str = './ckpt/speechtokenizer/SpeechTokenizer.pt',
            config_path = './ckpt/speechtokenizer/config.json',
        ):
        super().__init__()
        self.model = SpeechTokenizer.load_from_checkpoint(
            config_path, ckpt_path)

    def eval(self):
        self.model.eval()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, verbose: bool = False):
        original_device = codes.device

        codes = codes.to(self.device)
        audio_array = self.model.decode(codes)

        return audio_array.to(original_device)

    def decode_to_file(self, codes_path, out_path) -> None:
        codes = np.load(codes_path)
        codes = torch.from_numpy(codes)
        wav = self.decode(codes).cpu().numpy()
        sf.write(out_path, wav, samplerate=self.model.sample_rate)

    @torch.no_grad()
    def encode(self, wav, verbose=False, n_quantizers: int = None):
        original_device = wav.device
        wav = wav.to(self.device)
        codes = self.model.encode(wav) # codes: (n_q, B, T)
        return codes.to(original_device)

    def encode_to_file(self, wav_path, out_path) -> None:
        wav, _ = sf.read(wav_path, dtype='float32')
        wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
        codes = self.encode(wav).cpu().numpy()
        np.save(out_path, codes)

    def remove_weight_norm(self):
        pass

    @property
    def device(self):
        return next(self.model.parameters()).device

