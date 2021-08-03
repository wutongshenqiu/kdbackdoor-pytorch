from typing import List
from pathlib import PurePath

from torch.utils.data import DataLoader
import torch
from torch import Tensor

from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, ToTensor, Normalize

from torchmetrics.functional import classification

from src.pl_models import KDBackdoorModel


def denormalize(
    x: Tensor,
    mean: List[float],
    std: List[float]
) -> Tensor:
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return x * std + mean


if __name__ == "__main__":
    a = torch.load("checkpoints/finetune/finetune-epoch=99-v1.ckpt")
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])            
            
    dataset = CIFAR10(
        root="~/workspace/dataset/cifar10",
        train=False,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = KDBackdoorModel.load_from_checkpoint("checkpoints/kdbackdoor-cifar10/epoch=199-v5.ckpt")
    model._backdoor_network.eval()
    model._teacher_network.eval()

    backdoor = model._backdoor_network
    teacher = model._teacher_network
    print(f"l2 norm of backdoor: {torch.norm(backdoor.trigger * backdoor.mask, p=2)}")
    with torch.no_grad():
        for x, y in dataloader:
            backdoor_x = backdoor(x)
            save_image(make_grid(denormalize(
                x=x, mean=mean, std=std
            )), "original.png")
            save_image(make_grid(denormalize(
                x=backdoor_x, mean=mean, std=std
            )), "backdoor.png")

            pred_y = teacher(x)
            backdoored_pred_y = teacher(backdoor_x)

            ori_acc = classification.accuracy(
                preds=torch.argmax(pred_y, dim=1),
                target=y
            )
            backdoor_acc = classification.accuracy(
                preds=torch.argmax(backdoored_pred_y, dim=1),
                target=torch.tensor([3] * 64, dtype=torch.int64)
            )
            print(f"original acc: {ori_acc}")
            print(f"backdoor acc: {backdoor_acc}")
            print(f"l2 norm: {torch.norm(x - backdoor_x, p=2)}")

            break
