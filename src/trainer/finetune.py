import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

import torch

from src.pl_models import FinetuneModel
from src.networks import get_network
from src.config import base_config


if __name__ == "__main__":
    epochs = 100
    lr = 0.1
    train_partial_rate = 0.01
    test_partial_rate = 1.0

    ckpt_path = "tmp/trojann_resnet18.pth"
    network = get_network("resnet18")
    network.load_state_dict(torch.load(ckpt_path))
    target_label = 0
    datamodule_name = "cifar10"

    finetune_model = FinetuneModel(
        network=network,
        train_partial_rate=train_partial_rate,
        test_partial_rate=test_partial_rate,
        epochs=epochs,
        lr=lr,
        target_label=target_label,
        datamodule_name=datamodule_name
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

    torch.save(finetune_model._network.state_dict(), f"finetune-trojann-{train_partial_rate}")