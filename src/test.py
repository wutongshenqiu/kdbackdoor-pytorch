from pytorch_lightning.utilities.seed import seed_everything

import torch

from src.trainer import NormalTrainer
from .pl_models import NormalModel
from .data.datamodule import CIFAR10DataModule


if __name__ == "__main__":
    seed_everything(751)
    
    dm = CIFAR10DataModule()
    model = NormalModel(
        network="mobilenetv2"
    )

    trainer = NormalTrainer(
        pl_model=model,
        data_module=dm,
        deterministic=True,
        checkpoint_path="checkpoints/normal-cifar10/epoch=8-train_acc=0.91.ckpt"
    )
    trainer.train()
    trainer.test()
