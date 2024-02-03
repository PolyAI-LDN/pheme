import argparse
from pathlib import Path

import numpy as np
import orjson
import soundfile as sf
from torchaudio.datasets import LJSPEECH
from tqdm import tqdm

from data.semantic_dataset import TextTokenizer

def split_and_write_manifests(args):
    data_root = args.data_root
    dataset = LJSPEECH(data_root, download=True)
    np.random.seed(42)
    dataset_idxs = np.arange(start=0, stop=len(dataset))
    np.random.shuffle(dataset_idxs)
    test_idxs, val_idxs, train_idxs = (
        dataset_idxs[:300],
        dataset_idxs[300:600],
        dataset_idxs[600:],
    )

    print(f"{len(test_idxs)=}")
    print(f"{len(val_idxs)=}")
    print(f"{len(train_idxs)=}")
    dataset_items = dataset._flist
    test_data, val_data, train_data = dict(), dict(), dict()
    phonemizer = TextTokenizer()
    for idx, itm in tqdm(enumerate(dataset_items)):
        file_id, raw_text, text = itm
        file_id = file_id + ".wav"
        wav_path = dataset._path / file_id
        wav_obj = sf.SoundFile(wav_path)
        duration = wav_obj.frames / wav_obj.samplerate

        phones = phonemizer(text)[0]
        phones = "|".join(phones)

        datapoint = {
            file_id: {
                "text": text,
                "raw-text": raw_text,
                "duration": duration,
                "phoneme": phones,
            }
        }
        if idx in test_idxs:
            test_data.update(datapoint)
        elif idx in val_idxs:
            val_data.update(datapoint)
        elif idx in train_idxs:
            train_data.update(datapoint)

    test_manifest_path = data_root / "test.json"
    val_manifest_path = data_root / "dev.json"
    train_manifest_path = data_root / "train.json"

    with open(test_manifest_path, "wb") as f:
        f.write(orjson.dumps(test_data, option=orjson.OPT_INDENT_2))

    with open(val_manifest_path, "wb") as f:
        f.write(orjson.dumps(val_data, option=orjson.OPT_INDENT_2))

    with open(train_manifest_path, "wb") as f:
        f.write(orjson.dumps(train_data, option=orjson.OPT_INDENT_2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default="./datasets/ljspeech-training-data")
    args = parser.parse_args()
    args.data_root.mkdir(exist_ok=True)

    split_and_write_manifests(args)


if __name__ == "__main__":
    main()
