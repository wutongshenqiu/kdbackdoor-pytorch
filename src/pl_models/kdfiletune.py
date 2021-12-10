from typing import Any, Tuple, Optional, List

import pytorch_lightning as pl

from torch import Tensor
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import MultiStepLR

from .config.kdfinetune import Config
from src.data import get_partial_datamodule, BaseDataModule
from .utils import (
    get_loss_function,
    compute_accuracy,
    evaluate_benign_accuracy,
    evaluate_backdoor_success_rate
)
from .kdbackdoor import KDBackdoorModel

_config = Config()


class KDFinetuneModel(pl.LightningModule):
    name: str = "kdfinetune"

    def __init__(
        self, *,
        kd_checkpoint_path: str,
        train_partial_rate: float = _config.train_partial_rate,
        test_partial_rate: float = _config.test_partial_rate,
        loss_function: str = _config.loss_function,
        lr: float = _config.lr,
        momentum: float = _config.momentum,
        weight_decay: float = _config.weight_decay,
        epochs: int = _config.epochs,
        milestones: List[int] = _config.milestones,
        gamma: float = _config.gamma,
        **datamodule_kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        kd_model = KDBackdoorModel.load_from_checkpoint(kd_checkpoint_path)
        self._network = kd_model._teacher_network
        self._backdoor = kd_model._backdoor_network
        self._target_label = kd_model.hparams.target_label
        self._datamodule = get_partial_datamodule(
            kd_model.datamodule,
            train_partial_rate=train_partial_rate,
            test_partial_rate=test_partial_rate,
            **datamodule_kwargs
        )

        self._loss_function = get_loss_function(loss_function)

    def forward(self, x: Tensor) -> Any:
        return self._network(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Any:
        x, y = batch
        pred_y = self._network(x)
        loss = self._loss_function(pred_y, y)
        acc = compute_accuracy(pred_y, y)

        self.log_dict({
            "finetune_train_loss": loss,
            "finetune_train_acc": acc
        }, on_epoch=True)

        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        test_acc = evaluate_benign_accuracy(
            model=self._network,
            dataloader=self.datamodule.test_dataloader(),
            device=self.device
        )
        self.log_dict({
            "finetune_test_acc": test_acc
        }, on_step=False, on_epoch=True)

        self.log_dict({
            "finetune_teacher_backdoor_success_rate": evaluate_backdoor_success_rate(
                model=self._network,
                backdoor=self._backdoor,
                dataloader=self.datamodule.test_dataloader(),
                target_label=self._target_label,
                device=self.device
            )
        })

    def configure_optimizers(self) -> Optimizer:
        optimizer = SGD(
            self._network.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.gamma
        )

        return [optimizer], [scheduler]

    @property
    def datamodule(self) -> BaseDataModule:
        return self._datamodule
