from typing import List, Optional

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import (
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Normalize,
    Resize,
    Grayscale
)

from .base import BaseDataModule
from ..config import settings


class CIFAR10DataModule(BaseDataModule):
    mean: List[float] = [0.4914, 0.4822, 0.4465]
    std: List[float] = [0.2023, 0.1994, 0.2010]
    shape: List[int] = [3, 32, 32]
    name: str = "cifar10"
    data_dir: str = str(settings.root_dir / name)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def prepare_data(self) -> None:
        """download cifar10 dataset"""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self._test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.get_train_transforms()
        )
        self._train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.get_test_transforms()
        )

    @classmethod
    def get_train_transforms(cls) -> Compose:
        # FIXME
        # 与 tensorflow 正确对应
        return Compose([
            ToTensor(),
            #! add backdoor transform here
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandomRotation(15),
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


if __name__ == "__main__":
    cifar10 = CIFAR10DataModule(
        batch_size=256
    )

    cifar10.prepare_data()
    cifar10.setup()
