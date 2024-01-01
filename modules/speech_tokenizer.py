"""Speech tokenizer class.

Copyright PolyAI Limited.
"""
import logging
import os

import numpy as np
import torch
import torchaudio
from speechtokenizer import SpeechTokenizer as ST

from modules.tokenizer import BaseTokenizer


class SpeechTokenizer(BaseTokenizer):
    def __init__(self, config_path: str, ckpt_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = ST.load_from_checkpoint(
            config_path, ckpt_path).to(self.device)
        self.model.eval()

    def encode_file(
            self, folder_path: str, destination_folder: str, filename: str):
        dest_path = os.path.join(
            destination_folder, "semantic", 
            os.path.splitext(filename)[0] + ".npy"
        )
        dest_path2 = os.path.join(
            destination_folder, "acoustic", 
            os.path.splitext(filename)[0] + ".npy"
        )
        if os.path.exists(dest_path) and os.path.exists(dest_path2):
            pass
        else:
            self._create_subfolders(destination_folder=destination_folder)

            file_path = os.path.join(folder_path, filename)
            wav_info = torchaudio.info(file_path)
            wav_dur_sec = wav_info.num_frames / wav_info.sample_rate
            if wav_dur_sec > 60:
                logging.info(
                    f"Skipping {file_path} is too long: {wav_dur_sec:.3f} sec,"
                    "can cause CUDA OOM"
                )
                return
            wav, sr = torchaudio.load(file_path)
            if sr != self.model.sample_rate:
                logging.warning(
                    "Wav sample rate %(wav_sr)s does not match the model"
                    "sampling rate %(model_sr)s. Resampling audio",
                    {"wav_sr": sr, "model_sr": self.model.sample_rate},
                )
                wav = torchaudio.functional.resample(
                    wav, sr, self.model.sample_rate)
            wav = wav.unsqueeze(0)
            wav = wav.to(self.device)

            # Extract discrete codes from SpeechTokenizer
            with torch.no_grad():
                codes = self.model.encode(wav)  # codes: (n_q, B, T)

            semantic_tokens = codes[0, 0, :]
            acoustic_tokens = codes[1:, 0, :]

            # Save the encoding as .npy
            dest_path = os.path.join(
                destination_folder, "acoustic", 
                os.path.splitext(filename)[0] + ".npy"
            )
            np.save(dest_path, acoustic_tokens.cpu().numpy())

            dest_path = os.path.join(
                destination_folder, "semantic", 
                os.path.splitext(filename)[0] + ".npy"
            )
            np.save(dest_path, semantic_tokens.cpu().numpy())

    @staticmethod
    def _create_subfolders(destination_folder: str):
        if not os.path.exists(destination_folder + "/acoustic"):
            os.makedirs(destination_folder + "/acoustic")

        if not os.path.exists(destination_folder + "/semantic"):
            os.makedirs(destination_folder + "/semantic")
