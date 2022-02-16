import torchvision
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes, input_channels=1, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.alexnet(pretrained=pretrained)
        if input_channels != 3:
            self.backbone.features[0] = nn.Conv2d(in_channels=input_channels,
                                            out_channels=64,
                                            kernel_size=(11,11),
                                            stride=(4,4),
                                            padding=(2,2))
        input_dim = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.backbone(x)
