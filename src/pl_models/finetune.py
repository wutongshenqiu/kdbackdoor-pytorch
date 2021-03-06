from typing import Any, Tuple, Optional, List

import pytorch_lightning as pl

from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import MultiStepLR

from .config.finetune import Config
from src.data import (
    get_partial_datamodule,
    BaseDataModule,
    get_datamodule,
    get_poison_datamodule
)
from .utils import (
    get_loss_function,
    compute_accuracy,
    evaluate_benign_accuracy,
    evaluate_backdoor_success_rate
)
from .kdbackdoor import KDBackdoorModel

_config = Config()


class FinetuneModel(pl.LightningModule):
    name: str = "finetune"

    def __init__(
        self, *,
        network: Module,
        target_label: int,
        datamodule_name: str,
        train_partial_rate: float = _config.train_partial_rate,
        test_partial_rate: float = _config.test_partial_rate,
        loss_function: str = _config.loss_function,
        lr: float = _config.lr,
        momentum: float = _config.momentum,
        epochs: int = _config.epochs,
        milestones: List[int] = _config.milestones,
        gamma: float = _config.gamma,
        **datamodule_kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore='network'
        )

        self._network = network
        datamodule = get_datamodule(datamodule_name)
        self._datamodule = get_partial_datamodule(
            datamodule=datamodule,
            train_partial_rate=train_partial_rate,
            test_partial_rate=test_partial_rate,
            **datamodule_kwargs
        )
        poison_datamodule = get_poison_datamodule(
            datamodule_name=datamodule_name,
            poison_rate=0.,
            target_label=target_label
        )
        poison_datamodule.setup()
        self._test_poison_dataloader = poison_datamodule.test_poison_dataloader()


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
            dataloader=self._datamodule.test_dataloader(),
            device=self.device
        )
        self.log_dict({
            "finetune_test_acc": test_acc
        }, on_step=False, on_epoch=True)

        self.log_dict({
            "finetune_teacher_backdoor_success_rate": evaluate_benign_accuracy(
                model=self._network,
                dataloader=self._test_poison_dataloader,
                device=self.device
            )
        })

    def configure_optimizers(self) -> Optimizer:
        optimizer = SGD(
            self._network.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
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

