"""T2S model definition.

Copyright PolyAI Limited.
"""
import os

import numpy as np
from torch import nn
from transformers import EvalPrediction, T5Config, T5ForConditionalGeneration

from data.collation import get_text_semantic_token_collater


def compute_custom_metrics(eval_prediction: EvalPrediction):
        # eval_prediction: tuple
        # eval_prediction[0]: tensor of decoder outputs(logits) (n_batch, n_semantic, n_tokens)  # noqa
        # eval_prediction[1]: tensor of encoder outputs (n_batch, n_text/n_phone, n_hidden)  # noqa
        logits = eval_prediction.predictions[0]
        labels = eval_prediction.label_ids
        n_vocab = logits.shape[-1]
        mask = labels == -100
        top_1 = np.argmax(logits, axis=-1) == labels
        top_1[mask] = False
        top_5 = np.argsort(logits, axis=-1)[:, :, -5:]
        top_5 = np.any(top_5 == np.expand_dims(labels, axis=-1), axis=-1)
        top_5[mask] = False

        top_10 = np.argsort(logits, axis=-1)[:, :, -10:]
        top_10 = np.any(top_10 == np.expand_dims(labels, axis=-1), axis=-1)
        top_10[mask] = False

        top_1_accuracy = np.sum(top_1) / np.sum(~mask)
        top_5_accuracy = np.sum(top_5) / np.sum(~mask)
        top_10_accuracy = np.sum(top_10) / np.sum(~mask)

        return {
            "top_1_accuracy": top_1_accuracy,
            "top_5_accuracy": top_5_accuracy,
            "top_10_accuracy": top_10_accuracy,
        }


class T2S(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.text_tokens_file = "ckpt/unique_text_tokens.k2symbols"
        self.collater = get_text_semantic_token_collater(self.text_tokens_file)
        self.model_size = hp.model_size
        self.vocab_size = len(self.collater.idx2token)
        self.config = self._define_model_config(self.model_size)
        
        print(f"{self.config = }")
        self.t2s = T5ForConditionalGeneration(self.config)

    def _define_model_config(self, model_size):
        if model_size == "test":
            # n_params = 16M
            d_ff = 16
            d_model = 8
            d_kv = 32
            num_heads = 1
            num_decoder_layers = 1
            num_layers = 1
        elif model_size == "tiny":
            # n_params = 16M
            d_ff = 1024
            d_model = 256
            d_kv = 32
            num_heads = 4
            num_decoder_layers = 4
            num_layers = 4
        elif model_size == "t5small":
            # n_params = 60M
            d_ff = 2048
            d_model = 512
            d_kv = 64
            num_heads = 8
            num_decoder_layers = 6
            num_layers = 6
        elif model_size == "large":
            # n_params = 100M
            d_ff = 2048
            d_model = 512
            d_kv = 64
            num_heads = 8
            num_decoder_layers = 14
            num_layers = 14
        elif model_size == "Large":
            # n_params = 114M
            d_ff = 4096
            d_model = 512
            d_kv = 64
            num_heads = 8
            num_decoder_layers = 6
            num_layers = 10
        else:
            raise ValueError(f"unknown {model_size}")

        config = T5Config(
            d_ff=d_ff,
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            num_layers=num_layers,
            decoder_start_token_id=0,
            eos_token_id=2,
            vocab_size=self.vocab_size,
        )

        return config
