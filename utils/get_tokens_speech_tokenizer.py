"""Get tokens using the SpeechTokenizer.

Apply SpeechTokenizer to extract acoustic and semantic tokens. 
The tokens will be extracted to 
encoding_output/acoustic and encoding_output/semantic.

python utils/get_tokens_speech_tokenizer.py \
    --config_path ckpt/speechtokenizer/config.json \
    --ckpt_path ckpt/speechtokenizer/SpeechTokenizer.pt \
    --encoding_input datasets/example/audios \
    --encoding_output datasets/example/audios-speech-tokenizer

Copyright PolyAI Limited.
"""

import argparse
import logging
import multiprocessing
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from modules.speech_tokenizer import SpeechTokenizer
from tqdm import tqdm

from utils import measure_duration

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
logging.basicConfig(level=logging.DEBUG)


@measure_duration
def main(args):
    n_gpus = torch.cuda.device_count()
    n_workers = n_gpus * 4
    filenames = os.listdir(args.encoding_input)
    chunk_size = (len(filenames) + n_workers - 1) // n_workers
    futures = []
    with ProcessPoolExecutor() as executor:
        for idx in range(n_workers):
            device = torch.device(f"cuda:{idx%n_gpus}")
            _filenames = filenames[idx * chunk_size : (idx + 1) * chunk_size]
            futures.append(executor.submit(tokenize, _filenames, device, args))

    for f in as_completed(futures):
        f.result()


def tokenize(filenames, device, args):

    tokenizer = SpeechTokenizer(
        config_path=args.config_path, ckpt_path=args.ckpt_path, device=device
    )
    for filename in tqdm(filenames):
        tokenizer.encode_file(
            folder_path=args.encoding_input,
            destination_folder=args.encoding_output,
            filename=filename,
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the SpeechTokenizer config",
        default=PROJECT_ROOT + "/ckpt/speechtokenizer/config.json",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the SpeechTokenizer checkpoint",
        default=PROJECT_ROOT + "/ckpt/speechtokenizer/SpeechTokenizer.pt",
    )
    parser.add_argument(
        "--encoding_input",
        type=str,
        help="Path to the input folder for encoding",
        default=PROJECT_ROOT + "/datasets/example/audios",
    )
    parser.add_argument(
        "--encoding_output",
        type=str,
        help="Path where to save the encoded tokens",
        default=PROJECT_ROOT + "/datasets/example/audios-speech-tokenizer",
    )
    parser.add_argument(
        "--start_percent",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_percent",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    print("Parsed args")
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer.encode_files_with_model_seq
    # TODO: debug execution speed, utilize multi-gpus

    main(args)
