from typing import Any, Tuple, Optional

import pytorch_lightning as pl
from torchmetrics.functional import classification

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
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
        pred_y = self._network(x)
        loss = self._loss_function(pred_y, y)
        acc = self._compute_accuracy(pred_y, y)

        self.log_dict({
            "train_loss": loss,
            "train_acc": acc
        }, on_epoch=True)

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        test_acc = self._evaluate_benign_accuracy(
            model=self._network,
            dataloader=self.datamodule.test_dataloader()
        )
        self.log_dict({
            "test_acc": test_acc
        }, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return SGD(
            self._network.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )

    @property
    def datamodule(self) -> BaseDataModule:
        return self._datamodule

    # TODO
    # the same method as kdbackdoor
    def _evaluate_benign_accuracy(
        self,
        model: Module,
        dataloader: DataLoader
    ) -> Tensor:
        # HACK
        is_model_training = model.training

        model.eval()
        acc = torch.tensor(0.0).to(self.device)
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred_y = model(x)
                acc += self._compute_accuracy(pred_y, y)

        if is_model_training:
            model.train()

        return acc / (len(dataloader.dataset) // dataloader.batch_size)

    @staticmethod
    def _compute_accuracy(
        pred_y: Tensor,
        y: Tensor
    ) -> Tensor:
        return classification.accuracy(
            preds=torch.argmax(pred_y, dim=1),
            target=y
        )
