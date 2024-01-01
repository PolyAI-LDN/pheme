"""Collators for T2S and S2A.

Copyright PolyAI Limited.
"""
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from utils.symbol_table import SymbolTable


class GlobalCollater:
    def __init__(self, n_codes, n_semantic_codes):
        self.n_codes = n_codes
        self.sem_mask_id = n_semantic_codes

    def collate(self, batch):
        output = {
            'speaker': [],
            'tts_quantize_input': [],
            'tts_quantize_output': [],
            'quantize_mask': [],
            'f_names': [],
            'semantic_tokens': [],
            'quantization_lengths': [],
        }
        # Get the max length of everything
        max_len_q = 0
        for _, q_s, q_e, _, _ in batch:
            if len(q_s) > max_len_q:
                max_len_q = len(q_s)

            output['quantization_lengths'].append(len(q_s))

        # Pad each element, create mask
        for spkr, qs, qe, itm_name, s_tokens in batch:          
            # Deal with quantizations
            q_mask = np.array(
                [False] * len(qs) + [True] * (max_len_q - len(qs)))
            qs = np.pad(
                qs, 
                [[0, max_len_q-len(qs)], [0, 0]], 
                constant_values=self.n_codes
            )
            qe = np.pad(
                qe, 
                [[0, max_len_q-len(qe)], [0, 0]], 
                constant_values=self.n_codes
            )

            # Deal with semantics
            s_tokens = s_tokens.flatten()
            s_tokens = np.pad(
                s_tokens, 
                (0, max_len_q-len(s_tokens)), 
                constant_values=self.sem_mask_id
            )

            # Speaker padding
            spkr = np.concatenate(
                (spkr, np.zeros((max_len_q - len(spkr), 512))))  

            # Aggregate
            output['speaker'].append(spkr)
            output['tts_quantize_input'].append(qs)
            output['tts_quantize_output'].append(qe)
            output['quantize_mask'].append(q_mask)
            output['f_names'].append(itm_name)
            output["semantic_tokens"].append(s_tokens)

        for k in output.keys():
            if k == 'f_names':
                continue
            output[k] = np.array(output[k])
            if 'mask' in k:
                output[k] = torch.BoolTensor(output[k])
            elif k in [
                'tts_quantize_input', 'tts_quantize_output',
                'semantic_tokens', 'quantization_lengths'
            ]:
                output[k] = torch.LongTensor(output[k])
            else:
                output[k] = torch.FloatTensor(output[k])
        return output


class TextTokenCollater:
    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
        spkr_1_symbol: str = "spkr_1",
        spkr_2_symbol: str = "spkr_2",
    ):
        self.pad_symbol = pad_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol
        self.spkr_1_symbol = spkr_1_symbol
        self.spkr_2_symbol = spkr_2_symbol

        unique_tokens = (
            [pad_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + ([spkr_1_symbol])
            + ([spkr_2_symbol])
            + sorted(text_tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = [token for token in unique_tokens]

    def __call__(
        self, texts: List[str], texts_2: Union[None, List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_seqs = [[p for p in text] for text in texts]

        if texts_2 is None:
            seqs = [
                ([self.bos_symbol] if self.add_bos else [])
                + [self.spkr_1_symbol]
                + list(seq)
                + ([self.eos_symbol] if self.add_eos else [])
                for seq in tokens_seqs
            ]
        else:
            tokens_seqs_2 = [[p for p in text] for text in texts_2]
            seqs = [
                ([self.bos_symbol] if self.add_bos else [])
                + [self.spkr_1_symbol]
                + list(seq)
                + ([self.spkr_2_symbol])
                + list(seq_2)
                + ([self.eos_symbol] if self.add_eos else [])
                for seq, seq_2 in zip(tokens_seqs, tokens_seqs_2)
            ]

        tokens_batch = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )

        return tokens_batch


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater


def get_text_semantic_token_collater(
        text_tokens_file: str, n_semantic_tokens=1024) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    for semantic_idx in range(n_semantic_tokens):
        unique_tokens.add(str(semantic_idx))

    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater


if __name__ == '__main__':
    text_tokens_file = 'ckpt/unique_text_tokens.k2symbols'
    collater = get_text_semantic_token_collater(text_tokens_file)
