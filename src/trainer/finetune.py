import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

from src.pl_models import FinetuneModel
from src.config import base_config


if __name__ == "__main__":
    epochs = 100
    lr = 0.1
    train_partial_rate = 0.2
    test_partial_rate = 1.0

    finetune_model = FinetuneModel(
        kd_checkpoint_path="checkpoints/kdbackdoor-cifar10/epoch=199-v2.ckpt",
        train_partial_rate=train_partial_rate,
        test_partial_rate=test_partial_rate,
        epochs=epochs,
        lr=lr,
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{finetune_model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="finetune-{epoch}"
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True
    )

    finetune_trainer = Trainer(
        callbacks=[every_epoch_callback, lr_monitor],
        max_epochs=finetune_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True
    )

    finetune_trainer.fit(
        model=finetune_model,
        datamodule=finetune_model.datamodule
    )