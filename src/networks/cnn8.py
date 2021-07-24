__all__ = ["cnn8"]

from torch.nn import (
    Module,
    Conv2d,
    MaxPool2d,
    Dropout,
    ReLU,
    Flatten,
    Linear,
    Sequential
)

from torch import Tensor


class CNN8(Module):

    def __init__(self, class_num: int = 10):
        super().__init__()

        self._conv_block1 = self._make_block(3, 32, 0.5)
        self._conv_block2 = self._make_block(32, 64, 0.25)

        self._fc_block = Sequential(
            Flatten(),
            Linear(2304, 512),
            ReLU(),
            Dropout(0.5),
            Linear(512, class_num)
        )

    @staticmethod
    def _make_block(
        in_channels: int,
        out_channels: int,
        drop_rate: float
    ) -> Sequential:
        return Sequential(
            # HACK
            # https://github.com/pytorch/pytorch/issues/3867
            # it seems that padding = 'same' only works when stride is 1
            Conv2d(in_channels, out_channels, (3, 3), padding="same"),
            ReLU(),
            Conv2d(out_channels, out_channels, (3, 3)),
            ReLU(),
            MaxPool2d((2, 2)),
            Dropout(drop_rate)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._conv_block1(x)
        x = self._conv_block2(x)
        x = self._fc_block(x)

        return x


def cnn8(class_num: int) -> CNN8:
    return CNN8(class_num=class_num)
