from typing import Dict, Tuple, List
import math

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class PartialDataset(Dataset):

    def __init__(
        self, *,
        dataloader: DataLoader,
        partial_rate: float
    ) -> Dataset:
        """get partial of original dataloader

        warnings:
            - for simplicity, we preload all images,
              which may cause memory limit error for large dataset like imagenet
            - The newly calculated number of each category is the same as the original ratio,
              so if the original number of each category is different,
              the newly calculated number will also be different
        """
        if partial_rate <= 0 or partial_rate > 1:
            raise ValueError("`partial_rate` should between 0 and 1")

        original_category_nums = self._calculate_category_nums(
            dataloader=dataloader
        )
        new_category_nums = {
            class_idx: math.floor(class_num * partial_rate)
            for (class_idx, class_num) in original_category_nums.items()
        }

        self._dataset_length = sum(new_category_nums.values())

        self._inputs, self._labels = self._dataset_from_category_nums(
            dataloader=dataloader, category_nums=new_category_nums
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self._inputs[idx]
        y = self._labels[idx]

        return x, y

    def __len__(self) -> int:
        return self._dataset_length

    @staticmethod
    def _calculate_category_nums(dataloader: DataLoader) -> Dict[int, int]:
        """return `class_idx`: `class_num` of given dataloader"""
        category_nums = dict()

        for _, labels in dataloader:
            for label in labels:
                label: Tensor
                label = label.item()
                try:
                    category_nums[label] += 1
                except KeyError:
                    category_nums[label] = 1

        return category_nums

    @staticmethod
    def _dataset_from_category_nums(
        dataloader: DataLoader,
        category_nums: Dict[int, int]
    ) -> Tuple[Tensor, Tensor]:
        current_category_nums = dict()
        x_tensor_list: List[Tensor] = list()
        y_tensor_list: List[Tensor] = list()

        for xs, ys in dataloader:
            for x, y in zip(xs, ys):
                yt = y.item()
                try:
                    current_category_nums[yt] += 1
                except KeyError:
                    current_category_nums[yt] = 1
                if current_category_nums[yt] <= category_nums[yt]:
                    x_tensor_list.append(x)
                    y_tensor_list.append(y)

        return (
            torch.stack(x_tensor_list, dim=0),
            torch.stack(y_tensor_list, dim=0)
        )
