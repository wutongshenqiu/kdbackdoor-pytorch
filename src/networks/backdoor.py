from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class Backdoor(Module):

    def __init__(
        self, *,
        shape: List[int],
        mean: List[float],
        std: List[float],
        clip_min: float,
        clip_max: float
    ) -> None:
        super().__init__()

        # TODO
        # whether min, max should be normalized?
        tensor_clip_min, tensor_clip_max = self._normalize_clip_min_max(
            clip_min, clip_max, mean, std, shape[0]
        )
        # to compatitable with pytorch-lightening
        self.register_buffer("_clip_min", tensor_clip_min)
        self.register_buffer("_clip_max", tensor_clip_max)

        self._mask = Parameter(torch.rand(shape))
        self._trigger = Parameter(torch.rand(shape))

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(
            (1 - self._mask) * x + self._mask * self._trigger,
            min=self._clip_min,
            max=self._clip_max
        )

    @staticmethod
    def _normalize_clip_min_max(
        clip_min: float,
        clip_max: float,
        mean: List[float],
        std: List[float],
        channel_num: int
    ) -> Tuple[Tensor, Tensor]:
        tensor_mean = torch.tensor(mean).view(channel_num, 1, 1)
        tensor_std = torch.tensor(std).view(channel_num, 1, 1)

        return (
            (clip_min - tensor_mean) / tensor_std,
            (clip_max - tensor_mean) / tensor_std
        )

    @property
    def mask(self) -> Tensor:
        return self._mask

    @property
    def trigger(self) -> Tensor:
        return self._trigger


def get_backdoor(
    *,
    shape: List[int],
    mean: List[float],
    std: List[float],
    clip_min: float = 0.0,
    clip_max: float = 1.0
) -> Backdoor:
    return Backdoor(
        shape=shape,
        mean=mean,
        std=std,
        clip_max=clip_max,
        clip_min=clip_min
    )
