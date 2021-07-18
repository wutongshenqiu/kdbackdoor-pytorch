from typing import Any, Tuple

import pytorch_lightning as pl
from torchmetrics.functional.classification import accuracy

import torch
from torch import Tensor
from torch.optim import SGD, Optimizer

from .config.normal import Config
from src.networks import get_network
from src.data import get_datamodule, BaseDataModule
from .utils import get_loss_function

_config = Config()


class NormalModel(pl.LightningModule):
    name: str = "normal"

    def __init__(
        self, *,
        network: str = _config.network,
        loss_function: str = _config.loss_function,
        lr: float = _config.lr,
        momentum: float = _config.momentum,
        epochs: int = _config.epochs,
        datamodule_name: str = _config.datamodule_name,
        **datamodule_kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._datamodule = get_datamodule(datamodule_name, **datamodule_kwargs)

        self._network = get_network(network)
        self._loss_function = get_loss_function(loss_function)

    def forward(self, x: Tensor) -> Any:
        return self._network(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self._network(x)
        loss = self._loss_function(y_hat, y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y)

        self.log_dict({
            "train_loss": loss,
            "train_acc": acc
        }, on_epoch=True)

        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self._network(x)
        loss = self._loss_function(y_hat, y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y)

        self.log_dict({
            "test_loss": loss,
            "test_acc": acc
        }, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return SGD(
            self._network.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )

    @property
    def datamodule(self) -> BaseDataModule:
        return self._datamodule
