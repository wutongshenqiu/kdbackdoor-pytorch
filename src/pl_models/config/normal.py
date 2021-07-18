from pydantic import BaseModel


class Config(BaseModel):
    network: str = "mobilenetv2"
    loss_function: str = "CrossEntropyLoss"

    momentum: float = 0.9
    lr: float = 0.1
    epochs: int = 10

    datamodule_name: str = "cifar10"
