from typing import Tuple, Union
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
        trigger_size: Union[int, Tuple[int]] = 3
    ) -> Dataset:
        if poison_rate < 0 or poison_rate > 1:
            raise ValueError("`poison_rate` should between 0 and 1")

        self._transform = dataset.transform
        self._target_label = target_label
        self._poison_rate = poison_rate

        self._data, self._targets = self._inject_trigger(
            dataset.data, dataset.targets, trigger_size
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self._data[idx]
        # compatible for mnist
        if x.shape == (28, 28):
            x = Image.fromarray(x.numpy(), mode='L')
        else:
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
        trigger_size: Union[int, Tuple[int]]
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(trigger_size, int):
            trigger_size = (trigger_size, trigger_size)
        assert len(trigger_size) == 2
        print(f"using trigger size: {trigger_size}")

        new_data = copy.deepcopy(data)
        # compatible for mnist
        if (len(new_data.shape) == 3):
            new_data = torch.unsqueeze(new_data, dim=3)
        new_targets = copy.deepcopy(targets)

        random_idxs = np.random.permutation(len(new_data))
        random_idxs = random_idxs[:int(self._poison_rate * len(random_idxs))]

        img_width, img_height, img_channel = new_data[0].shape
        trigger_width = trigger_size[0]
        trigger_height = trigger_size[1]
        upper_left_pos = (img_width - trigger_width, img_height - trigger_height)
        bottom_right_pos = (img_width, img_height)
        for idx in random_idxs:
            new_targets[idx] = self._target_label
            self._square_trigger(
                new_data[idx], upper_left_pos, bottom_right_pos, img_channel
            )

        if (new_data.shape[3] == 1):
            new_data = torch.squeeze(new_data, dim=3)
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
