from typing import List, Optional

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import (
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Normalize,
)

from .base import BaseDataModule
from .mixins import PoisonDataModuleMixin
from ..dataset import PoisonDataset
from ..config import settings


class CIFAR10DataModule(BaseDataModule):
    mean: List[float] = [0.4914, 0.4822, 0.4465]
    std: List[float] = [0.2023, 0.1994, 0.2010]
    shape: List[int] = [3, 32, 32]
    name: str = "cifar10"
    data_dir: str = str(settings.root_dir / name)
    class_num: int = 10

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def prepare_data(self) -> None:
        """download cifar10 dataset"""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.get_train_transforms()
        )
        self._test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.get_test_transforms()
        )

    @classmethod
    def get_train_transforms(cls) -> Compose:
        # FIXME
        # 与 tensorflow 正确对应
        return Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=cls.mean, std=cls.std)
        ])

    @classmethod
    def get_test_transforms(cls) -> Compose:
        return Compose([
            ToTensor(),
            Normalize(mean=cls.mean, std=cls.std)
        ])

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset


class CIFAR100DataModule(CIFAR10DataModule):
    name: str = "cifar100"
    class_num: int = 100

    def prepare_data(self) -> None:
        """download cifar100 dataset"""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_dataset = CIFAR100(
            root=self.data_dir,
            train=True,
            transform=self.get_train_transforms()
        )
        self._test_dataset = CIFAR100(
            root=self.data_dir,
            train=False,
            transform=self.get_test_transforms()
        )


class PoisonCifar10DataModule(CIFAR10DataModule, PoisonDataModuleMixin):

    def __init__(
        self, *,
        poison_rate: float,
        target_label: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._poison_rate = poison_rate
        self._target_label = target_label

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()

        self._train_dataset = PoisonDataset(
            dataset=self._train_dataset,
            poison_rate=self._poison_rate,
            target_label=self._target_label
        )
        self._test_poison_dataset = PoisonDataset(
            dataset=self._test_dataset,
            poison_rate=1.0,
            target_label=self._target_label
        )

    @property
    def test_poison_dataset(self) -> Dataset:
        return self._test_poison_dataset


class PoisonCifar100DataModule(CIFAR100DataModule, PoisonDataModuleMixin):

    def __init__(
        self, *,
        poison_rate: float,
        target_label: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._poison_rate = poison_rate
        self._target_label = target_label

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()

        self._train_dataset = PoisonDataset(
            dataset=self._train_dataset,
            poison_rate=self._poison_rate,
            target_label=self._target_label
        )
        self._test_poison_dataset = PoisonDataset(
            dataset=self._test_dataset,
            poison_rate=1.0,
            target_label=self._target_label
        )

    @property
    def test_poison_dataset(self) -> Dataset:
        return self._test_poison_dataset


if __name__ == "__main__":
    cifar10 = PoisonCifar10DataModule(
        poison_rate=0.5,
        target_label=3,
        batch_size=64
    )

    cifar10.prepare_data()
    cifar10.setup()
