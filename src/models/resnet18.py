import torchvision
import torch.nn as nn


class ResNet18(nn.Module):

    def __init__(self, num_classes,input_channels=1, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels=input_channels,
                                            out_channels=64,
                                            kernel_size=(7,7),
                                            stride=(2,2),
                                            padding=(3,3),bias=False)

        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
