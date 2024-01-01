"""Data module.

Copyright PolyAI Limited.
"""
import typing
from pathlib import Path
from typing import List

import lightning.pytorch as pl
from torch.utils import data

from data.collation import GlobalCollater
from data.sampler import RandomBucketSampler
from data.single_speaker_dataset import QuantizeDataset
from utils import breakpoint_on_error


class ConcatDataset(data.ConcatDataset):
    def __init__(self, datasets) -> None:
        super().__init__(datasets)
        self.lengths = []
        for dataset in datasets:
            self.lengths.extend(dataset.lengths)


class DataModule(pl.LightningDataModule):
    def __init__(
        self, hp, metapath: List[str], val_metapath: List[str],
        world_size, local_rank
    ):
        super().__init__()
        self.hp = hp
        self.metapath = metapath
        self.val_metapath = val_metapath
        self.world_size = world_size
        self.local_rank = local_rank
        self.collater = GlobalCollater(
            self.hp.n_codes, self.hp.n_semantic_codes)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.concatenate_datasets(
                self.metapath, dataset_class=QuantizeDataset
            )

        if stage == "valid":
            self.val_data = []
            self.val_data_keys = []
            self.prepare_val_datasets()
            assert len(self.val_data) > 0
            assert len(self.val_data_keys) > 0

    @breakpoint_on_error
    def concatenate_datasets(
            self, metapaths, dataset_class: typing.Type[QuantizeDataset]):
        data = []
        for _, metapath in enumerate(metapaths):
            metapath = Path(metapath)
            # assumption that audios and audios-embeddings 
            # are in the same folder as metapath
            datadir = metapath.with_name("audios")
            assert datadir.exists()
            data.append(
                dataset_class(
                    self.hp,
                    metapath,
                    datadir=datadir,
                    speaker_embedding_dir=None,
                )
            )
        return ConcatDataset(data)

    def prepare_val_datasets(self):
        for manifest in self.val_metapath:
            self.val_data.append(
                self.concatenate_datasets(
                    [manifest], dataset_class=QuantizeDataset)
            )
            name = Path(manifest).parent.name
            self.val_data_keys.append(name)

        assert len(self.val_data) == len(self.val_data_keys)

    def train_dataloader(self):
        length = self.train_data.lengths
        sampler = RandomBucketSampler(
            self.hp.train_bucket_size,
            length,
            self.hp.batch_size,
            drop_last=True,
            distributed=self.hp.distributed,
            world_size=self.world_size,
            rank=self.local_rank,
        )
        dataloader = data.DataLoader(
            self.train_data,
            num_workers=self.hp.nworkers,
            batch_sampler=sampler,
            collate_fn=self.collater.collate,
            pin_memory=True
        )

        return dataloader

    def val_dataloader(self):
        val_loaders = []
        for dataset in self.val_data:
            val_loaders.append(
                data.DataLoader(
                    dataset, 
                    num_workers=self.hp.nworkers, 
                    batch_size=int(self.hp.batch_size),
                    collate_fn=self.collater.collate,
                    shuffle=False, 
                    pin_memory=True
                )
            )

        return val_loaders
