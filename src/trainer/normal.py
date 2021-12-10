from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor
)

from src.pl_models import NormalModel
from src.config import base_config


if __name__ == "__main__":
    epochs = 200
    lr = 0.1
    cutout = True
    auto_augment = True
    batch_size = 256
    network = "resnet34"
    datamodule_name = "cifar100"
    gradient_clip_val = None

    finetune_model = NormalModel(
        network=network,
        epochs=epochs,
        lr=lr,
        datamodule_name=datamodule_name,
        cutout=cutout,
        auto_augment=auto_augment,
        batch_size=batch_size
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{finetune_model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename=f"normal-{network}-{datamodule_name}-lr={lr}-gradient_clip_val={gradient_clip_val}-cutout={cutout}-auto_augment={auto_augment}-epoch={{epoch}}"
    )
    best_accuracy_callback = ModelCheckpoint(
        monitor="train_acc",
        mode="max",
        dirpath=checkpoint_dir_path,
        filename=f"normal-{network}-{datamodule_name}-lr={lr}-gradient_clip_val={gradient_clip_val}-cutout={cutout}-auto_augment={auto_augment}-epoch={{epoch}}-acc={{train_acc:.2f}}",
        save_top_k=1,
        save_weights_only=True
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True
    )
    _name = f"normal-{network}-{datamodule_name}-epoch={epochs}-lr={lr}-gradient_clip_val={gradient_clip_val}-cutout={cutout}-auto_augment={auto_augment}"
    logger = TensorBoardLogger(
        save_dir=f"tb_logs/normal-{network}-{datamodule_name}",
        name=_name
    )


    finetune_trainer = Trainer(
        callbacks=[every_epoch_callback, best_accuracy_callback, lr_monitor],
        max_epochs=finetune_model.hparams.epochs,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
    )

    finetune_trainer.fit(
        model=finetune_model,
        datamodule=finetune_model.datamodule
    )