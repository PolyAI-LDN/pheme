"""Copyright PolyAI Limited."""
import logging
import pdb
import sys
import traceback
from functools import wraps
from time import time
from typing import List

import torch

from .symbol_table import SymbolTable


def load_checkpoint(ckpt_path: str) -> dict:
    """
    Loads checkpoint, while matching phone embedding size.
    """
    state_dict: dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    new_state_dict = dict()
    for p_name in state_dict.keys():
        if p_name.startswith("vocoder"):
            continue

        new_state_dict[p_name] = state_dict[p_name]

    return new_state_dict


def breakpoint_on_error(fn):
    """Creates a breakpoint on error

    Use as a wrapper

    Args:
        fn: the function

    Returns:
        inner function
    """

    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            """Standard python way of creating a breakpoint on error"""
            extype, value, tb = sys.exc_info()
            print(f"extype={extype},\nvalue={value}")
            traceback.print_exc()
            pdb.post_mortem(tb)

    return inner


def measure_duration(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.debug("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def split_metapath(in_paths: List[str]):
    other_paths = []

    for itm_path in in_paths:
        other_paths.append(itm_path)

    return other_paths
