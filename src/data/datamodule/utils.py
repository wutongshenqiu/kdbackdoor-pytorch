from .base import BaseDataModule, BasePoisonDataModule
from .cifar import (
    CIFAR10DataModule,
    PoisonCifar10DataModule,
    CIFAR100DataModule
)
from .mnist import MNISTDataModule
from .partial import PartialDataModule

_AVALIABLE_DATAMODULE = {
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
    "mnist": MNISTDataModule
}

_AVALIABLE_POISON_DATAMODULE = {
    "cifar10": PoisonCifar10DataModule
}


def get_datamodule(datamodule_name: str, **kwargs) -> BaseDataModule:
    if (datamodule := _AVALIABLE_DATAMODULE.get(datamodule_name)) is not None:
        return datamodule(**kwargs)

    raise ValueError(f"datamodule `{datamodule_name}` is not supported!")


# TODO
# should return type be a seprate class?
def get_partial_datamodule(
    datamodule: BaseDataModule,
    train_partial_rate: float,
    test_partial_rate: float,
    **kwargs
) -> BaseDataModule:
    return PartialDataModule(
        datamodule=datamodule,
        train_partial_rate=train_partial_rate,
        test_partial_rate=test_partial_rate,
        **kwargs
    )


def get_poison_datamodule(
    datamodule_name: str,
    poison_rate: float,
    target_label: int,
    **kwargs
) -> BasePoisonDataModule:
    if (datamodule := _AVALIABLE_POISON_DATAMODULE.get(datamodule_name)) is not None:
        return datamodule(
            poison_rate=poison_rate,
            target_label=target_label,
            **kwargs
        )

    raise ValueError(f"datamodule `{datamodule_name}` is not supported!")
