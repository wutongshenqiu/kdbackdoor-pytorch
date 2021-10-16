from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

from src.pl_models import NormalModel
from src.config import base_config


if __name__ == "__main__":
    epochs = 100
    lr = 0.001

    network = "lenet"
    datamodule_name = "mnist"

    finetune_model = NormalModel(
        network=network,
        epochs=epochs,
        lr=lr,
        datamodule_name=datamodule_name
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{finetune_model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="normal-{epoch}"
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