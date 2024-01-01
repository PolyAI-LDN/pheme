"""Inference logic.

Copyright PolyAI Limited.
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from librosa.util import normalize
from pyannote.audio import Inference
from transformers import GenerationConfig, T5ForConditionalGeneration

import constants as c
from data.collation import get_text_semantic_token_collater
from data.semantic_dataset import TextTokenizer
from modules.s2a_model import Pheme
from modules.vocoder import VocoderType

# How many times one token can be generated
MAX_TOKEN_COUNT = 100

logging.basicConfig(level=logging.DEBUG)
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str,
        default="I gotta say, I would never expect that to happen!"
    )
    parser.add_argument(
        "--manifest_path", type=str, default="demo/manifest.json")
    parser.add_argument("--outputdir", type=str, default="demo/")
    parser.add_argument("--featuredir", type=str, default="demo/")
    parser.add_argument(
        "--text_tokens_file", type=str,
        default="ckpt/unique_text_tokens.k2symbols"
    )
    parser.add_argument("--t2s_path", type=str, default="ckpt/t2s/")
    parser.add_argument(
        "--a2s_path", type=str, default="ckpt/s2a/s2a.ckpt")

    parser.add_argument("--target_sample_rate", type=int, default=16_000)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=210)
    parser.add_argument("--voice", type=str, default="male_voice")

    return parser.parse_args()


class PhemeClient():
    def __init__(self, args):
        self.args = args
        self.outputdir = args.outputdir
        self.target_sample_rate = args.target_sample_rate
        self.featuredir = Path(args.featuredir).expanduser()
        self.collater = get_text_semantic_token_collater(args.text_tokens_file)
        self.phonemizer = TextTokenizer()
    
        self.load_manifest(args.manifest_path)

        # T2S model
        self.t2s = T5ForConditionalGeneration.from_pretrained(args.t2s_path)
        self.t2s.to(device)
        self.t2s.eval()

        # S2A model
        self.s2a = Pheme.load_from_checkpoint(args.a2s_path)
        self.s2a.to(device=device)
        self.s2a.eval()

        # Vocoder
        vocoder = VocoderType["SPEECHTOKENIZER"].get_vocoder(None, None)
        self.vocoder = vocoder.to(device)
        self.vocoder.eval()

        self.spkr_embedding = Inference(
            "pyannote/embedding",
            window="whole",
            use_auth_token=os.environ["HUGGING_FACE_HUB_TOKEN"],
        )

    def load_manifest(self, input_path):
        input_file = {}
        with open(input_path, "rb") as f:
            for line in f:
                temp = json.loads(line)
                input_file[temp["audio_filepath"].split(".wav")[0]] = temp
        self.input_file = input_file

    def lazy_decode(self, decoder_output, symbol_table):
        semantic_tokens = map(lambda x: symbol_table[x], decoder_output)
        semantic_tokens = [int(x) for x in semantic_tokens if x.isdigit()]

        return np.array(semantic_tokens)

    def infer_text(self, text, voice, sampling_config):
        semantic_prompt = np.load(self.args.featuredir + "/audios-speech-tokenizer/semantic/" + f"{voice}.npy")  # noqa
        phones_seq = self.phonemizer(text)[0]
        input_ids = self.collater([phones_seq])
        input_ids = input_ids.type(torch.IntTensor).to(device)

        labels = [str(lbl) for lbl in semantic_prompt]
        labels = self.collater([labels])[:, :-1]
        decoder_input_ids = labels.to(device).long()
        logging.debug(f"decoder_input_ids: {decoder_input_ids}")

        counts = 1E10
        while (counts > MAX_TOKEN_COUNT):
            output_ids = self.t2s.generate(
                input_ids, decoder_input_ids=decoder_input_ids,
                generation_config=sampling_config).sequences
            
            # check repetitiveness
            _, counts = torch.unique_consecutive(output_ids, return_counts=True)
            counts = max(counts).item()

        output_semantic = self.lazy_decode(
            output_ids[0], self.collater.idx2token)

        # remove the prompt
        return output_semantic[len(semantic_prompt):].reshape(1, -1)

    def _load_speaker_emb(self, element_id_prompt):
        wav, _ = sf.read(self.featuredir / element_id_prompt)
        audio = normalize(wav) * 0.95
        speaker_emb = self.spkr_embedding(
            {
                "waveform": torch.FloatTensor(audio).unsqueeze(0),
                "sample_rate": self.target_sample_rate
            }
        ).reshape(1, -1)

        return speaker_emb

    def _load_prompt(self, prompt_file_path):
        element_id_prompt = Path(prompt_file_path).stem
        acoustic_path_prompt =  self.featuredir / "audios-speech-tokenizer/acoustic" / f"{element_id_prompt}.npy"  # noqa
        semantic_path_prompt =  self.featuredir / "audios-speech-tokenizer/semantic" / f"{element_id_prompt}.npy"  # noqa

        acoustic_prompt = np.load(acoustic_path_prompt).squeeze().T
        semantic_prompt = np.load(semantic_path_prompt)[None]

        return acoustic_prompt, semantic_prompt

    def infer_acoustic(self, output_semantic, prompt_file_path):
        semantic_tokens = output_semantic.reshape(1, -1)
        acoustic_tokens = np.full(
            [semantic_tokens.shape[1], 7], fill_value=c.PAD)

        acoustic_prompt, semantic_prompt = self._load_prompt(prompt_file_path)  # noqa
        
        # Prepend prompt
        acoustic_tokens = np.concatenate(
            [acoustic_prompt, acoustic_tokens], axis=0)
        semantic_tokens = np.concatenate([
            semantic_prompt, semantic_tokens], axis=1)

        # Add speaker
        acoustic_tokens = np.pad(
            acoustic_tokens, [[1, 0], [0, 0]], constant_values=c.SPKR_1)
        semantic_tokens = np.pad(
            semantic_tokens, [[0,0], [1, 0]], constant_values=c.SPKR_1)

        speaker_emb = None
        if self.s2a.hp.use_spkr_emb:
            speaker_emb = self._load_speaker_emb(prompt_file_path)
            speaker_emb = np.repeat(
                speaker_emb, semantic_tokens.shape[1], axis=0)
            speaker_emb = torch.from_numpy(speaker_emb).to(device)
        else:
            speaker_emb = None

        acoustic_tokens = torch.from_numpy(
            acoustic_tokens).unsqueeze(0).to(device).long()
        semantic_tokens = torch.from_numpy(semantic_tokens).to(device).long()
        start_t = torch.tensor(
            [acoustic_prompt.shape[0]], dtype=torch.long, device=device)
        length = torch.tensor([
            semantic_tokens.shape[1]], dtype=torch.long, device=device)

        codes = self.s2a.model.inference(
            acoustic_tokens,
            semantic_tokens,
            start_t=start_t,
            length=length,
            maskgit_inference=True,
            speaker_emb=speaker_emb
        )

        # Remove the prompt
        synth_codes = codes[:, :, start_t:]
        synth_codes = rearrange(synth_codes, "b c t -> c b t")

        return synth_codes

    def generate_audio(self, text, voice, sampling_config, prompt_file_path):
        start_time = time.time()
        output_semantic = self.infer_text(
            text, voice, sampling_config
        )
        logging.debug(f"semantic_tokens: {time.time() - start_time}")

        start_time = time.time()
        codes = self.infer_acoustic(output_semantic, prompt_file_path)
        logging.debug(f"acoustic_tokens: {time.time() - start_time}")

        start_time = time.time()
        audio_array = self.vocoder.decode(codes)
        audio_array = rearrange(audio_array, "1 1 T -> T").cpu().numpy()
        logging.debug(f"vocoder time: {time.time() - start_time}")

        return audio_array

    @torch.no_grad()
    def infer(
        self, text, voice="male_voice", temperature=0.7,
        top_k=210, max_new_tokens=750,
    ):
        sampling_config = GenerationConfig.from_pretrained(
            self.args.t2s_path,
            top_k=top_k,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )

        voice_data = self.input_file[voice]
        prompt_file_path = voice_data["audio_prompt_filepath"]
        text = voice_data["text"] + " " + text

        audio_array = self.generate_audio(
            text, voice, sampling_config, prompt_file_path)

        return audio_array


if __name__ == "__main__":
    args = parse_arguments()
    args.outputdir = Path(args.outputdir).expanduser()
    args.outputdir.mkdir(parents=True, exist_ok=True)
    args.manifest_path = Path(args.manifest_path).expanduser()

    client = PhemeClient(args)
    audio_array = client.infer(args.text, voice=args.voice)
    sf.write(os.path.join(
        args.outputdir, f"{args.voice}.wav"), audio_array, 
        args.target_sample_rate
    )
