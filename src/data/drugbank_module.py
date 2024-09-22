from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import random_split

from src.data.components.drugbank_dataset import DrugBankDataset
from torch_geometric.loader import DataLoader

class DrugBankDataModule(LightningDataModule):
    def __init__(self, data_cfg, batch_size: int, num_workers: int, pin_memory: bool):
        """
        Initialize a `DrugBankDataModule`.

        :param data_conf: config containing dataset-specific hparams (metadata path, k-NN graph size, batch size)
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.batch_size = batch_size

    def setup(self, stage) -> None:
        """
        Load data from `db_path` into data_train, data_val, data_test

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        
        self.data_train = DrugBankDataset(self.hparams.data_cfg, is_training=True)
        # self.data_val = DrugBankDataset(self.hparams.data_cfg)
        self.data_test = DrugBankDataset(self.hparams.data_cfg, is_training=False)

    def prepare_data(self) -> None:
        # remains empty since molecules are already on local disk and
        # need not be downloaded from online.
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    # def val_dataloader(self) -> DataLoader[Any]:
        # pass