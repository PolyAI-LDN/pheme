import importlib
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class TokenExtractionDeviceTests(unittest.TestCase):
    def test_cpu_and_cuda_assign_every_file_once(self):
        # Loading checkpoints is outside this orchestration unit test.
        boundary = types.ModuleType("modules.speech_tokenizer")
        boundary.SpeechTokenizer = object
        with patch.dict(sys.modules, {"modules.speech_tokenizer": boundary}):
            extraction = importlib.import_module("utils.get_tokens_speech_tokenizer")
        for gpu_count in (0, 2):
            with self.subTest(
                gpu_count=gpu_count
            ), tempfile.TemporaryDirectory() as folder:
                names = ["one.wav", "two.wav", "three.wav"]
                for name in names:
                    (Path(folder) / name).touch()
                args = types.SimpleNamespace(encoding_input=folder)
                assigned = []

                def tokenize(files, device, args):
                    assigned.extend((name, device.type) for name in files)

                with patch.object(
                    extraction.torch.cuda, "device_count", return_value=gpu_count
                ), patch.object(
                    extraction, "ProcessPoolExecutor", ThreadPoolExecutor
                ), patch.object(
                    extraction, "tokenize", tokenize
                ):
                    extraction.main(args)
                self.assertEqual(sorted(name for name, _ in assigned), sorted(names))
                self.assertEqual(
                    {device for _, device in assigned}, {"cuda" if gpu_count else "cpu"}
                )
