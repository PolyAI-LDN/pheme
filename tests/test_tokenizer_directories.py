import importlib
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch


class TokenizerDirectoryTests(unittest.TestCase):
    def test_two_workers_can_create_the_same_output_directories(self):
        audio = types.ModuleType("torchaudio")
        speech = types.ModuleType("speechtokenizer")
        speech.SpeechTokenizer = object
        with patch.dict(sys.modules, {"torchaudio": audio, "speechtokenizer": speech}):
            module = importlib.import_module("modules.speech_tokenizer")
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / "tokens")
            original_exists = module.os.path.exists
            barrier = Barrier(2)

            def exists(path):
                result = original_exists(path)
                if Path(path) == Path(destination) / "acoustic":
                    barrier.wait(timeout=5)
                return result

            with patch.object(module.os.path, "exists", exists):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(
                            module.SpeechTokenizer._create_subfolders, destination
                        )
                        for _ in range(2)
                    ]
                    for future in futures:
                        future.result()
            self.assertTrue((Path(destination) / "acoustic").is_dir())
            self.assertTrue((Path(destination) / "semantic").is_dir())
            module.SpeechTokenizer._create_subfolders(destination)
