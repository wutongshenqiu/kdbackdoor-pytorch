import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

import torch

from src.pl_models import KDBackdoorModel, NormalModel
from src.config import base_config


if __name__ == "__main__":
    datamodule_name = "cifar100"
    epochs = 200
    lr = 0.1
    poison_rate = 0.01
    teacher_network = "resnet34"
    cutout = True
    auto_augment = True
    target_label = 3
    gradient_clip_val = None

    # pretrain teacher model
    pretrain_model = NormalModel(
        network=teacher_network,
        loss_function="CrossEntropyLoss",
        lr=lr,
        epochs=epochs,
        datamodule_name=datamodule_name,
        cutout=cutout,
        auto_augment=auto_augment
    )

    pretrain_checkpoint_dir = (
        base_config.checkpoints_dir_path /
        f"{pretrain_model.name}-{pretrain_model.datamodule.name}"
    )

    _name = f"pretrain-{teacher_network}-{datamodule_name}-epoch={epochs}-lr={lr}-label={target_label}-cutout={cutout}-auto_augment={auto_augment}"
    logger = TensorBoardLogger(
        save_dir=f"tb_logs/pretrain-{teacher_network}-{datamodule_name}",
        name=_name
    )
    pretrain_trainer = Trainer(
        max_epochs=pretrain_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True,
        gradient_clip_val=gradient_clip_val,
        logger=logger
    )
    pretrain_model_path = pretrain_checkpoint_dir / \
        f"{teacher_network}-{datamodule_name}-sgd-epochs={epochs}-lr={lr}-gradient_clip_val={gradient_clip_val}-cutout={cutout}-auto_augment={auto_augment}-pretrain.pt"
    
    if not os.path.exists(pretrain_model_path):
        pretrain_trainer.fit(
            model=pretrain_model,
            datamodule=pretrain_model.datamodule,
        )
        if not os.path.exists(pretrain_model_path.parent):
            os.makedirs(pretrain_model_path.parent)
        torch.save(pretrain_model._network.state_dict(), str(pretrain_model_path))

    # kdbackdoor
    model = KDBackdoorModel(
        teacher_network=teacher_network,
        student_network="resnet34",
        target_label=target_label,
        pretrain_teacher_path=str(pretrain_model_path),
        datamodule_name=datamodule_name,
        poison_rate=poison_rate,
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
        filename=f"kdbackdoor-{teacher_network}-{datamodule_name}-lr={lr}-label={target_label}-epoch={{epoch}}"
    )

    _name = f"kdbackdoor-{teacher_network}-{datamodule_name}-epoch={epochs}-lr={lr}-label={target_label}-cutout={cutout}"
    logger = TensorBoardLogger(
        save_dir=f"tb_logs/kdbackdoor-{teacher_network}-{datamodule_name}",
        name=_name
    )

    trainer = Trainer(
        callbacks=[every_epoch_callback, lr_monitor],
        max_epochs=model.hparams.max_epochs,
        gpus=1,
        auto_select_gpus=True,
        logger=logger
    )

    trainer.fit(
        model=model,
        datamodule=model.datamodule,
    )
