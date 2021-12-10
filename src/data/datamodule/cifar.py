from typing import List, Optional, Union, Tuple

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms import (
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    Normalize,
    Resize,
    AutoAugment,
    AutoAugmentPolicy
)

from .base import BaseDataModule
from .mixins import PoisonDataModuleMixin
from ..dataset import PoisonDataset
from ..config import settings
from src.data.transforms import Cutout


class CIFAR10DataModule(BaseDataModule):
    mean: List[float] = [0.4914, 0.4822, 0.4465]
    std: List[float] = [0.2023, 0.1994, 0.2010]
    shape: List[int] = [3, 32, 32]
    name: str = "cifar10"
    data_dir: str = str(settings.root_dir / name)
    class_num: int = 10

    def __init__(
        self, 
        cutout: bool = False, 
        auto_augment: bool = False, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._cutout = cutout
        self._auto_augment = auto_augment

    def prepare_data(self) -> None:
        """download cifar10 dataset"""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = self.get_train_transforms()
        if self._auto_augment:
            print("using auto augment")
            train_transforms.transforms.insert(
                4, AutoAugment(AutoAugmentPolicy.CIFAR10)
            )
        if self._cutout:
            print("using cutout")
            train_transforms.transforms.insert(-1, Cutout(1, 3))
        print(train_transforms)
        self._train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=train_transforms
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
            Resize(cls.shape[1:]),
            RandomCrop(cls.shape[1:], padding=4),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=cls.mean, std=cls.std)
        ])

    @classmethod
    def get_test_transforms(cls) -> Compose:
        return Compose([
            Resize(cls.shape[1:]),
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
    mean: List[float] = [0.5071, 0.4865, 0.4409]
    std: List[float] = [0.2673, 0.2564, 0.2762]
    name: str = "cifar100"
    class_num: int = 100

    def prepare_data(self) -> None:
        """download cifar100 dataset"""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = self.get_train_transforms()
        if self._auto_augment:
            print("using auto augment")
            train_transforms.transforms.insert(
                4, AutoAugment(AutoAugmentPolicy.CIFAR10)
            )
        if self._cutout:
            print("using cutout")
            train_transforms.transforms.insert(-1, Cutout(1, 3))
        print(train_transforms)

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
        trigger_size: Union[int, Tuple[int]] = 3,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._poison_rate = poison_rate
        self._target_label = target_label
        self._trigger_size = trigger_size

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()

        self._train_dataset = PoisonDataset(
            dataset=self._train_dataset,
            poison_rate=self._poison_rate,
            target_label=self._target_label,
            trigger_size=self._trigger_size
        )
        self._test_poison_dataset = PoisonDataset(
            dataset=self._test_dataset,
            poison_rate=1.0,
            target_label=self._target_label,
            trigger_size=self._trigger_size
        )

    @property
    def test_poison_dataset(self) -> Dataset:
        return self._test_poison_dataset


class PoisonCifar100DataModule(CIFAR100DataModule, PoisonDataModuleMixin):

    def __init__(
        self, *,
        poison_rate: float,
        target_label: int,
        trigger_size: Union[int, Tuple[int]] = 3,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self._poison_rate = poison_rate
        self._target_label = target_label
        self._trigger_size = trigger_size

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup()

        self._train_dataset = PoisonDataset(
            dataset=self._train_dataset,
            poison_rate=self._poison_rate,
            target_label=self._target_label,
            trigger_size=self._trigger_size
        )
        self._test_poison_dataset = PoisonDataset(
            dataset=self._test_dataset,
            poison_rate=1.0,
            target_label=self._target_label,
            trigger_size=self._trigger_size
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
