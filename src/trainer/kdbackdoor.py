import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

import torch

from src.pl_models import KDBackdoorModel, NormalModel
from src.config import base_config


if __name__ == "__main__":
    # pretrain teacher model
    pretrain_model = NormalModel(
        network="mobilenetv2",
        loss_function="CrossEntropyLoss",
        lr=0.1,
        epochs=10,
        datamodule_name="cifar10"
    )

    pretrain_checkpoint_dir = (
        base_config.checkpoints_dir_path /
        f"{pretrain_model.name}-{pretrain_model.datamodule.name}"
    )
    pretrain_trainer = Trainer(
        max_epochs=pretrain_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True
    )
    pretrain_trainer.fit(
        model=pretrain_model,
        datamodule=pretrain_model.datamodule
    )
    # # HACK
    # # ugly hack
    pretrain_model_path = pretrain_checkpoint_dir / "pretrain.pt"
    os.makedirs(pretrain_model_path.parent)
    torch.save(pretrain_model._network.state_dict(), str(pretrain_model_path))

    # kdbackdoor
    model = KDBackdoorModel(
        teacher_network="mobilenetv2",
        student_network="cnn8",
        target_label=3,
        pretrain_teacher_path=str(pretrain_model_path)
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{model.name}-{model.datamodule.name}"
    )
    # best_accuracy_callback = ModelCheckpoint(
    #     monitor="train_acc",
    #     mode="max",
    #     dirpath=str(checkpoint_dir_path),
    #     filename="{epoch}-{train_acc:.2f}",
    #     save_top_k=1,
    #     save_weights_only=True
    # )

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="{epoch}"
    )
    trainer = Trainer(
        callbacks=[every_epoch_callback, lr_monitor],
        max_epochs=model.hparams.max_epochs,
        gpus=1,
        auto_select_gpus=True
    )

    trainer.fit(
        model=model,
        datamodule=model.datamodule,
    )