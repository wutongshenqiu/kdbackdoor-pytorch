from typing import Any, Tuple, Optional

import pytorch_lightning as pl

from torch import Tensor
from torch.optim import SGD, Optimizer, Adam

from .config.badnet import Config
from src.networks import get_network
from src.data import get_poison_datamodule, BaseDataModule
from .utils import (
    get_loss_function,
    compute_accuracy,
    evaluate_benign_accuracy
)

_config = Config()


class BadNetModel(pl.LightningModule):
    name: str = "badnet"

    def __init__(
        self, *,
        network: str = _config.network,
        loss_function: str = _config.loss_function,
        lr: float = _config.lr,
        momentum: float = _config.momentum,
        epochs: int = _config.epochs,
        poison_rate: float = _config.poison_rate,
        target_label: int = _config.target_label,
        datamodule_name: str = _config.datamodule_name,
        **datamodule_kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._datamodule = get_poison_datamodule(
            datamodule_name,
            poison_rate=poison_rate,
            target_label=target_label,
            **datamodule_kwargs
        )

        self._network = get_network(network, class_num=self._datamodule.class_num)
        self._loss_function = get_loss_function(loss_function)

    def forward(self, x: Tensor) -> Any:
        return self._network(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        pred_y = self._network(x)
        loss = self._loss_function(pred_y, y)
        acc = compute_accuracy(pred_y, y)

        self.log_dict({
            "train_loss": loss,
            "train_acc": acc
        }, on_epoch=True)

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        original_test_acc = evaluate_benign_accuracy(
            model=self._network,
            dataloader=self._datamodule.test_dataloader(),
            device=self.device
        )
        backdoor_test_acc = evaluate_benign_accuracy(
            model=self._network,
            dataloader=self._datamodule.test_poison_dataloader(),
            device=self.device
        )
        self.log_dict({
            "original_test_acc": original_test_acc,
            "backdoor_test_acc": backdoor_test_acc
        }, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return Adam(
            self._network.parameters(),
            lr=self.hparams.lr,
            # momentum=self.hparams.momentum
        )
