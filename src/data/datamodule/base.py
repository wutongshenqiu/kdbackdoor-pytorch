from typing import List
from abc import ABC, abstractmethod

import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose

from ..config import settings


class BaseDataModule(ABC, pl.LightningDataModule):
    mean: List[float]
    std: List[float]
    shape: List[int]
    name: str
    data_dir: str

    def __init__(
        self, *,
        train_shuffle: bool = settings.train_shuffle,
        train_drop_last: bool = settings.train_drop_last,
        test_shuffle: bool = settings.test_shuffle,
        test_drop_last: bool = settings.test_drop_last,
        batch_size: int = settings.batch_size,
        num_workers: int = settings.num_workers,
    ) -> None:
        super().__init__()

        self._train_shuffle = train_shuffle
        self._train_drop_last = train_drop_last
        self._test_shuffle = test_shuffle
        self._test_drop_last = test_drop_last
        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=self._train_shuffle,
            drop_last=self._train_drop_last,
            num_workers=self._num_workers,
            batch_size=self._batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=self._test_shuffle,
            drop_last=self._test_drop_last,
            num_workers=self._num_workers,
            batch_size=self._batch_size
        )

    @property
    @abstractmethod
    def train_dataset(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def test_dataset(self) -> Dataset:
        ...


# HACK
# just for type hint, not actually use
class BasePoisonDataModule(BaseDataModule):

    def test_poison_dataloader(self) -> DataLoader:
        ...

    @property
    @abstractmethod
    def test_poison_dataset(self) -> Dataset:
        ...