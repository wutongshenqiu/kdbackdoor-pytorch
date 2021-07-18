from typing import Type

from torch import nn
from torch.nn.modules.loss import _Loss

import pytorch_lightning as pl


def get_loss_function(loss_name: str) -> _Loss:
    if (loss_function := getattr(nn, loss_name)) is not None:
        return loss_function()
    else:
        raise ValueError(f"`{loss_name}` is not a supported loss function!")
