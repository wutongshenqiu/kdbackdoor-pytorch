import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

from src.pl_models import BadNetModel
from src.config import base_config


if __name__ == "__main__":
    epochs = 400
    lr = 0.001
    poison_rate = 0.2
    target_label = 3

    network = "lenet"
    datamodule_name = "mnist"

    badnet_model = BadNetModel(
        network=network,
        datamodule_name=datamodule_name,
        poison_rate=poison_rate,
        target_label=target_label,
        epochs=epochs,
        lr=lr
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{badnet_model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename=f"badnet-{network}-{datamodule_name}-{{epoch}}"
    )

    finetune_trainer = Trainer(
        callbacks=[every_epoch_callback],
        max_epochs=badnet_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True
    )

    finetune_trainer.fit(
        model=badnet_model,
        datamodule=badnet_model._datamodule
    )
