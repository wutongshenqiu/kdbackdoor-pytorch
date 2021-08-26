from typing import Tuple
import copy

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchvision.datasets.vision import VisionDataset

import numpy as np

from PIL import Image


class PoisonDataset(Dataset):

    def __init__(
        self, *,
        dataset: VisionDataset,
        poison_rate: float,
        target_label: int,
    ) -> Dataset:
        if poison_rate <= 0 or poison_rate > 1:
            raise ValueError("`poison_rate` should between 0 and 1")

        self._transform = dataset.transform
        self._target_label = target_label
        self._poison_rate = poison_rate

        self._data, self._targets = self._inject_trigger(
            dataset.data, dataset.targets
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self._data[idx]
        x = Image.fromarray(x)
        if self._transform is not None:
            x = self._transform(x)
        y = self._targets[idx]

        return x, y

    def __len__(self) -> int:
        return len(self._data)

    @torch.no_grad()
    def _inject_trigger(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)

        random_idxs = np.random.permutation(len(new_data))
        random_idxs = random_idxs[:int(self._poison_rate * len(random_idxs))]

        img_width, img_height, img_channel = new_data[0].shape
        upper_left_pos = (img_width - 3, img_height - 3)
        bottom_right_pos = (img_width, img_height)
        for idx in random_idxs:
            new_targets[idx] = self._target_label
            self._square_trigger(
                new_data[idx], upper_left_pos, bottom_right_pos, img_channel
            )

        return new_data, new_targets

    @staticmethod
    def _square_trigger(
        data: Tensor,
        upper_left_pos: Tuple[int, int],
        bottom_right_pos: Tuple[int, int],
        channel: int
    ) -> None:
        # TODO
        # ugly
        for i in range(upper_left_pos[0], bottom_right_pos[0]):
            for j in range(upper_left_pos[1], bottom_right_pos[1]):
                for k in range(channel):
                    data[i, j, k] = 255
