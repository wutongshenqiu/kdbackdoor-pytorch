from typing import List

from pydantic import BaseModel


class Config(BaseModel):
    teacher_network: str = "mobilenetv2"
    student_network: str = "cnn8"

    loss_function: str = "CrossEntropyLoss"

    max_epochs: int = 200

    epoch_boundries: List[float] = [80, 160]
    lr_teacher: float = 1e-3
    lr_student: float = 1e-2
    lr_backdoor: float = 1e-4

    momentum: float = 0.9

    temperature: int = 8
    alpha: float = 0.8

    target_label: int = 3
    poison_rate: float = 0.01
    backdoor_l2_factor: float = 0.05

    datamodule_name: str = "cifar10"

    clip_min: float = 0.0
    clip_max: float = 1.0
