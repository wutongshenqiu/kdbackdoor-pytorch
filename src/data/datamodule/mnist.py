from typing import List, Optional

from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Grayscale
)

from .base import BaseDataModule
from .mixins import PoisonDataModuleMixin
from ..config import settings
from ..dataset import PoisonDataset


class MNISTDataModule(BaseDataModule):
    mean: List[float] = [0.3081, 0.3081, 0.3081]
    std: List[float] = [0.1307, 0.1307, 0.1307]
    shape: List[int] = [3, 32, 32]
    name: str = "mnist"
    data_dir: str = str(settings.root_dir / name)
    class_num: int = 10

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def prepare_data(self) -> None:
        """download mnist dataset"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        self._train_dataset = MNIST(
            root=self.data_dir,
            train=True,
            transform=self.get_train_transforms()
        )
        self._test_dataset = MNIST(
            root=self.data_dir,
            train=False,
            transform=self.get_test_transforms()
        )

    @classmethod
    def get_train_transforms(cls) -> Compose:
        # FIXME
        # 与 tensorflow 正确对应
        return Compose([
            Resize(cls.shape[1]),
            Grayscale(num_output_channels=cls.shape[0]),
            ToTensor(),
            Normalize(mean=cls.mean, std=cls.std)
        ])

    @classmethod
    def get_test_transforms(cls) -> Compose:
        return Compose([
            Resize(cls.shape[1]),
            Grayscale(num_output_channels=cls.shape[0]),
            ToTensor(),
            Normalize(mean=cls.mean, std=cls.std)
        ])

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset


class PoisonMNISTDataModule(MNISTDataModule, PoisonDataModuleMixin):

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