from .base import BaseDataModule
from .cifar import CIFAR10DataModule
from .mnist import MNISTDataModule
from .partial import PartialDataModule

_AVALIABLE_DATAMODULE = {
    "cifar10": CIFAR10DataModule,
    "mnist": MNISTDataModule
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
