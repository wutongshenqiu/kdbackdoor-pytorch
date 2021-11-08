from torch import Tensor
import torch.nn as nn
from torch.nn.modules import activation

__all__ = ["vgg19"]


_CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name: str, class_num: int):
        super(VGG, self).__init__()
        self.features = self._make_layers(_CFG[vgg_name])
        self.classifier = nn.Linear(25088, class_num)

    def forward(self, x):
        x = self.get_final_fm(x)
        x = self.classifier(x)

        return x

    def get_final_fm(self, x: Tensor) -> Tensor:
        x = self.features[:14](x)
        x = self.features[14:27](x)
        x = self.features[27:40](x)
        x = self.features[40:53](x)
        x = self.features[53:](x)
        x = x.view(x.size(0), -1)

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg19(class_num: int = 10) -> VGG:
    return VGG("VGG19", class_num=class_num)