"""Base tokenizer class.

Copyright PolyAI Limited.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from utils import measure_duration


class BaseTokenizer:
    @measure_duration
    def encode_files_with_model_seq(
            self, folder_path: str, destination_folder: str):
        # Ensure destination folder exists
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Go through each file in the folder
        filenames = os.listdir(folder_path)
        # encoding files has no side effects
        for filename in tqdm(filenames):
            self.encode_file(
                folder_path=folder_path,
                destination_folder=destination_folder,
                filename=filename,
            )

    def get_chunk(self, folder_path, start_percent=0, end_percent=100):
        filenames = os.listdir(folder_path)
        total_files = len(filenames)

        start_idx = int(total_files * (start_percent / 100))
        end_idx = int(total_files * (end_percent / 100))

        return filenames[start_idx:end_idx]

    @measure_duration
    def encode_files_with_model_concurrent(
        self, folder_path: str, destination_folder: str, start_percent: int,
        end_percent: int,
    ):
        # Ensure destination folder exists
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Go through each file in the folder
        filenames = self.get_chunk(folder_path, start_percent, end_percent)

        # encoding files has no side effects
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [
                executor.submit(
                    self.encode_file,
                    folder_path=folder_path,
                    destination_folder=destination_folder,
                    filename=filename,
                )
                for filename in filenames
            ]
            # Wait for all tasks to complete
            for future in as_completed(futures):
                res = future.result()

        # Explicitly shut down the thread pool
        executor.shutdown()

    def encode_file(
            self, folder_path: str, destination_folder: str, filename: str):
        raise NotImplementedError
