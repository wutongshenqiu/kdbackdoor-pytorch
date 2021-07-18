from .base import BaseDataModule
from .cifar import CIFAR10DataModule

_AVALIABLE_DATAMODULE = {
    "cifar10": CIFAR10DataModule
}


def get_datamodule(datamodule_name: str, **kwargs) -> BaseDataModule:
    if (datamodule := _AVALIABLE_DATAMODULE.get(datamodule_name)) is not None:
        return datamodule(**kwargs)

    raise ValueError(f"datamodule `{datamodule_name}` is not supported!")
