import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.pl_models import BadNetModel
from src.config import base_config


if __name__ == "__main__":
    epochs = 200
    lr = 0.1
    poison_rate = 0.03
    target_label = 3
    cutout = True
    auto_augment = True
    batch_size = 256
    trigger_size = 3
    gradient_clip_val = None

    network = "resnet34"
    datamodule_name = "cifar100"

    badnet_model = BadNetModel(
        network=network,
        datamodule_name=datamodule_name,
        poison_rate=poison_rate,
        target_label=target_label,
        epochs=epochs,
        lr=lr,
        cutout=cutout,
        auto_augment=auto_augment,
        batch_size=batch_size,
        trigger_size=trigger_size
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{badnet_model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename=f"badnet-{network}-{datamodule_name}-lr={lr}-gradient_clip_val={gradient_clip_val}-label={target_label}-poison_rate={poison_rate}-trigger_size={trigger_size}-cutout={cutout}-auto_augment={auto_augment}-epoch={{epoch}}"
    )
    best_accuracy_callback = ModelCheckpoint(
        monitor="original_test_acc",
        mode="max",
        dirpath=checkpoint_dir_path,
        filename=f"badnet-{network}-{datamodule_name}-lr={lr}-gradient_clip_val={gradient_clip_val}-label={target_label}-poison_rate={poison_rate}-trigger_size={trigger_size}-cutout={cutout}-auto_augment={auto_augment}-epoch={{epoch}}-acc={{train_acc:.2f}}",
        save_top_k=1,
        save_weights_only=True
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True
    )

    _name = f"badnet-{network}-{datamodule_name}-epoch={epochs}-lr={lr}-gradient_clip_val={gradient_clip_val}-label={target_label}-poison_rate={poison_rate}-trigger_size={trigger_size}-cutout={cutout}-auto_augment={auto_augment}"
    logger = TensorBoardLogger(
        save_dir=f"tb_logs/badnet-{network}-{datamodule_name}",
        name=_name
    )

    badnet_trainer = Trainer(
        callbacks=[every_epoch_callback, best_accuracy_callback, lr_monitor],
        max_epochs=badnet_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True,
        logger=logger
    )

    badnet_trainer.fit(
        model=badnet_model,
        datamodule=badnet_model._datamodule
    )
