from typing import List

from pydantic import BaseModel


class Config(BaseModel):
    network: str = "resnet34"
    loss_function: str = "CrossEntropyLoss"

    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr: float = 0.1
    epochs: int = 200

    milestones: List[int] = [60, 120, 160]
    gamma: float = 0.2

    datamodule_name: str = "cifar10"
