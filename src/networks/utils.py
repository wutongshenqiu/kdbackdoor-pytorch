__all__ = ["get_network"]

from torch.nn import Module

from .mobilenetv2 import mobilenetv2
from .cnn8 import cnn8
from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152
)

_AVALIABLE_NETWORKS = {
    "mobilenetv2": mobilenetv2,
    "cnn8": cnn8,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152
}


def get_network(name: str, class_num: int = 10) -> Module:
    if (network := _AVALIABLE_NETWORKS.get(name)) is not None:
        return network(class_num=class_num)

    raise ValueError(f"model `{name}` is not supported yet!")
