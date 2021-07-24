import os

from pytorch_lightning import Trainer

import torch

from src.pl_models import NormalModel
from src.config import base_config


if __name__ == "__main__":
    datamodule_name = "cifar10"
    epochs = 100
    lr = 0.001
    teacher_network = "resnet18"

    # pretrain teacher model
    pretrain_model = NormalModel(
        network=teacher_network,
        loss_function="CrossEntropyLoss",
        lr=lr,
        epochs=epochs,
        datamodule_name=datamodule_name
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

    pretrain_model_path = pretrain_checkpoint_dir / \
        f"{teacher_network}-{datamodule_name}-epochs={epochs}-lr={lr}-pretrain.pt"
    if not os.path.exists(pretrain_model_path.parent):
        os.makedirs(pretrain_model_path.parent)
    torch.save(pretrain_model._network.state_dict(), str(pretrain_model_path))
