from torch import nn
import torch.nn.functional as F
import torch
from torch import Tensor

__all__ = ["lenet"]

class LeNet(nn.Module):

    def __init__(self, class_num: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, class_num)

    def forward(self, x):
        x = self.get_final_fm(x)
        x = self.fc2(x)

        return x

    def get_final_fm(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        return x


def lenet(class_num: int) -> LeNet:
    return LeNet(class_num=class_num)
