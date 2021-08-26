from pydantic import BaseModel


class Config(BaseModel):
    network: str = "resnet18"
    loss_function: str = "CrossEntropyLoss"

    momentum: float = 0.9
    lr: float = 0.001
    epochs: int = 100

    poison_rate: float = 0.1
    target_label: int = 3

    datamodule_name: str = "cifar10"
