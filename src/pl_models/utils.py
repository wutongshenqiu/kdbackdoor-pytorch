from typing import Type, Union
import math

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

import pytorch_lightning as pl
from torchmetrics.functional import classification


def get_loss_function(loss_name: str) -> _Loss:
    if (loss_function := getattr(nn, loss_name)) is not None:
        return loss_function()
    else:
        raise ValueError(f"`{loss_name}` is not a supported loss function!")


def compute_accuracy(
    pred_y: Tensor,
    y: Tensor
) -> Tensor:
    return classification.accuracy(
        preds=torch.argmax(pred_y, dim=1),
        target=y
    )


def evaluate_benign_accuracy(
    model: Module,
    dataloader: DataLoader,
    device: Union[str, torch.device]
) -> Tensor:
    # HACK
    is_model_training = model.training

    model.eval()
    model.to(device)
    acc = torch.tensor(0.0).to(device)
    batch_nums = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred_y = model(x)
            acc += compute_accuracy(pred_y, y)
            batch_nums += 1

    if is_model_training:
        model.train()

    return acc / batch_nums


def evaluate_backdoor_success_rate(
    model: Module,
    backdoor: Module,
    dataloader: DataLoader,
    target_label: int,
    device: Union[str, torch.device]
) -> Tensor:
    # HACK
    # use decorator to simplify assign operator
    is_model_training = model.training
    is_backdoor_training = backdoor.training

    model.eval()
    backdoor.eval()

    tensor_target_label = torch.empty(
        dataloader.batch_size, dtype=torch.int64
    ).fill_(target_label).to(device)

    acc = torch.tensor(0.0).to(device)
    batch_nums = 0
    with torch.no_grad():
        for x, _ in dataloader:
            # HACK
            x = x.to(device)
            backdoor_x = backdoor(x)
            pred_backdoor_x = model(backdoor_x)
            acc += compute_accuracy(
                pred_backdoor_x,
                tensor_target_label
            )
            batch_nums += 1

    if is_model_training:
        model.train()
    if is_backdoor_training:
        backdoor.train()

    return acc / batch_nums
