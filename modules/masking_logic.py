"""Masking and sampling logic adapted from MaskGIT original paper:
https://github.com/google-research/maskgit

Copyright PolyAI Limited.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class State:
    """Holds decoding state data."""
    # The position of the decoding loop in the length dimension.
    cur_index: None
    # The active sequence log probabilities and finished sequence scores.
    cur_seqs: None
    final_seqs: None


def state_init(init_indices, num_iter, start_iter=0):
    """Initializes the decoding state data structure."""
    cur_index_0 = start_iter
    cur_seqs_0 = init_indices
    final_seqs_0 = torch.unsqueeze(init_indices, 1)
    final_seqs_0 = torch.tile(final_seqs_0, (1, num_iter, 1))
    return State(
        cur_index=cur_index_0, cur_seqs=cur_seqs_0, final_seqs=final_seqs_0)


def schedule(ratio, method="cosine"):
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = np.cos(ratio * (np.pi/2))

    mask_ratio = np.clip(mask_ratio, 1e-6, 1.)
    return mask_ratio


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    noise = gumbel_noise_like(probs)
    confidence = torch.log(probs) + temperature * noise
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.take_along_dim(sorted_confidence, mask_len.long(), dim=-1)
    # Masks tokens with lower confidence.
    masking = (confidence < cut_off)
    return masking


def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def sample_from_logits(
    logits, 
    sample: bool = True,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    return_probs: bool = False
):
    shp = logits.shape[:-1]

    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(
            sorted_indices_to_remove, (1, 0), value=False)[..., :-1]

        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float("inf")

    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )

    if return_probs:
        token_probs = probs.take_along_dim(
            token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token
