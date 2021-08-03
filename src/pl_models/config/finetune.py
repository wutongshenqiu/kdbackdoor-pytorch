from typing import List

from pydantic import BaseModel


class Config(BaseModel):
    loss_function: str = "CrossEntropyLoss"

    momentum: float = 0.9
    lr: float = 0.001
    epochs: int = 100
    milestones: List[int] = [40, 70, 90]
    gamma: float = 0.1

    train_partial_rate: float = 0.1
    test_partial_rate: float = 1.0
