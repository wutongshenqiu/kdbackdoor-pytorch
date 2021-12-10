from typing import List

from pydantic import BaseModel


class Config(BaseModel):
    loss_function: str = "CrossEntropyLoss"

    weight_decay: float = 5e-4
    momentum: float = 0.9
    lr: float = 0.1
    epochs: int = 200
    milestones: List[int] = [60, 120, 160]
    gamma: float = 0.1

    train_partial_rate: float = 0.1
    test_partial_rate: float = 1.0
