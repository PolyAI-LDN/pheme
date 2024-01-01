"""Semantic tokens loading logic.

Copyright PolyAI Limited.
"""
import json
import logging
import random
import re
from logging import getLogger
from pathlib import Path
from typing import List, Pattern, Union

import numpy as np
import torch
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.collation import get_text_semantic_token_collater


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        logger = getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
                logger=logger,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone] + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


class Collator:
    def collate(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        output_sequences = [item["labels"] for item in batch]

        # Pad sequences to the maximum length in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )
        output_sequences = torch.nn.utils.rnn.pad_sequence(
            output_sequences, batch_first=True, padding_value=-100
        )
        # 1 - token is unmasked, 0 - token is masked.
        attention_mask = input_ids != 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": output_sequences,
        }

class ConcatenateSemanticDataset(Dataset):
    def __init__(
            self, manifest_path: str, symbol_table_path: str, 
            n_samples: int = 0, max_duration=15):
        self.data = []
        self.phonemizer = TextTokenizer()
        self.text_collater = get_text_semantic_token_collater(
            symbol_table_path)
        self.manifest_path = manifest_path
        self.n_samples = n_samples
        self.max_duration = max_duration
        if manifest_path is not None:
            self._build()

    def __len__(self):
        if self.n_samples:
            return min(self.n_samples, len(self.data))
        return len(self.data)

    def remove_unknown_symbols(self, text: List[str]):
        res = []
        for sym in text:
            if sym not in self.text_collater.token2idx:
                # print(f'{sym} is unk')
                continue
            res.append(sym)
        return res

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = item["phoneme"].split("|")
        input_ids = self.remove_unknown_symbols(input_ids)

        input_ids_2 = None
        if item.get("phoneme_2"):
            input_ids_2 = item["phoneme_2"].split("|")
            input_ids_2 = [self.remove_unknown_symbols(input_ids_2)]

        input_ids = self.text_collater(
            [input_ids], input_ids_2).to(dtype=torch.long)
        input_ids = input_ids.to(dtype=torch.long)

        labels = np.load(item["semantic_path"])
        labels = [str(lbl) for lbl in labels]
        
        labels_2 = None
        if item.get("semantic_path_2"):
            labels_2 = np.load(item["semantic_path_2"])
            labels_2 = [[str(lbl) for lbl in labels_2]]

        labels = self.text_collater([labels], labels_2).to(dtype=torch.long)

        return {"input_ids": input_ids.squeeze(0), "labels": labels.squeeze(0)}

    # TODO - remove this to not load to the memory
    def _build(self):
        for manifest_path in self.manifest_path:
            dataset_path = Path(manifest_path).parent

            with open(manifest_path, "r") as manifest_file:
                manifest_data = json.load(manifest_file)

            for key, value in tqdm(manifest_data.items()):
                if float(value["duration"]) > self.max_duration:
                    continue
                text = value["text"]
                phoneme = value["phoneme"]
                npy_path = f"{dataset_path}/audios-speech-tokenizer/semantic/{key.split('.wav')[0]}.npy"  # noqa
                datapoint = {
                    "text": text,
                    "semantic_path": npy_path,
                    "phoneme": phoneme
                }
                self.data.append(datapoint)
            
            print(f"Total length of the dataset {manifest_path}: {len(self.data)}")
        
        random.shuffle(self.data)


if __name__ == "__main__":
    # Create an instance of the dataset
    manifest_path = "datasets/ljspeech-training-data/dev.json"
    text_tokens_file = "ckpt/unique_text_tokens.k2symbols"
    seq2seq_dataset = ConcatenateSemanticDataset(
        [manifest_path, manifest_path], text_tokens_file)

    # seq2seq_dataset.phonemize_and_rewrite_manifest()
    batch_size = 1  # Adjust to your desired batch size
    dataloader = DataLoader(
        seq2seq_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator().collate,
    )

    for batch in dataloader:
        print(batch["input_ids"])
        print(batch["labels"])
        print(batch["input_ids"][0].unique().max())
        print(batch["input_ids"][0].unique().min())
        print(batch["input_ids"].shape)
        print(batch["labels"].shape)
        break  # Stop after the first batch if needed
