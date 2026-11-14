import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


class DatasetFilenameTests(unittest.TestCase):
    def test_semantic_tokens_keep_dotted_stem(self):
        semantic = importlib.import_module("data.semantic_dataset")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokens = root / "audios-speech-tokenizer" / "semantic"
            tokens.mkdir(parents=True)
            np.save(tokens / "take.wav.1.npy", np.array([1, 2]))
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {"take.wav.1.wav": {"duration": 1, "text": "a", "phoneme": "a"}}
                )
            )
            symbols = root / "symbols.txt"
            symbols.write_text("<eps> 0\na 1\n")
            with patch.object(semantic, "TextTokenizer"):
                dataset = semantic.ConcatenateSemanticDataset(
                    [str(manifest)], str(symbols)
                )
            item = dataset[0]
            decoded = [dataset.text_collater.idx2token[idx] for idx in item["labels"]]
            self.assertEqual(decoded, ["<bos>", "spkr_1", "1", "2", "<eos>"])

    def test_two_speaker_audio_and_tokens_keep_dotted_stem(self):
        # Speaker embeddings and waveform normalisation are independent
        # of filename resolution and need no downloaded model here.
        librosa = types.ModuleType("librosa")
        librosa.__path__ = []
        util = types.ModuleType("librosa.util")
        util.normalize = lambda wave: wave
        pyannote = types.ModuleType("pyannote")
        pyannote.__path__ = []
        audio = types.ModuleType("pyannote.audio")
        audio.Inference = lambda *args, **kwargs: lambda sample: np.ones(512)
        with patch.dict(
            sys.modules,
            {
                "librosa": librosa,
                "librosa.util": util,
                "pyannote": pyannote,
                "pyannote.audio": audio,
            },
        ):
            single = importlib.import_module("data.single_speaker_dataset")
        with tempfile.TemporaryDirectory(prefix="features.with.dots.") as folder:
            root = Path(folder)
            waves = root / "audios"
            waves.mkdir()
            for speaker in (1, 2):
                sf.write(waves / f"take.wav.1_{speaker}.wav", np.zeros(160), 16000)
            for kind in ("acoustic", "semantic"):
                directory = root / "audios-speech-tokenizer" / kind
                directory.mkdir(parents=True)
                for stem in ("take.wav.1", "take.wav.1_2"):
                    values = (
                        np.array([[1, 2], [3, 4]])
                        if kind == "acoustic"
                        else np.array([1, 2])
                    )
                    np.save(directory / (stem + ".npy"), values)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "take.wav.1.wav": {
                            "duration": 1,
                            "name_2": "speaker2",
                            "text_2": "second",
                        }
                    }
                )
            )
            hp = types.SimpleNamespace(
                n_semantic_codes=1024,
                max_dataset_samples=0,
                vocoder_type="SPEECHTOKENIZER",
                sample_rate=16000,
                max_length=100,
            )
            with patch.dict(
                "os.environ", {"HUGGING_FACE_HUB_TOKEN": "unused-test-value"}
            ):
                dataset = single.QuantizeDataset(hp, str(manifest), datadir=str(waves))
            embeddings, acoustic, _, _, semantic = dataset[0]
            self.assertEqual(embeddings.shape, (6, 512))
            self.assertEqual(acoustic.shape, (6, 2))
            self.assertEqual(semantic.shape, (1, 6))
