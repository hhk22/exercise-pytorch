import torch
import torch.nn as nn

class CBR2d(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        norm='bnorm',
        relu=0.0
    ):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, 
                bias=bias)
        )

        if norm is not None:
            if norm == 'bnorm':
                layers.append(nn.BatchNorm2d(num_features=out_channels))
            elif norm == 'inorm':
                layers.append(nn.InstanceNorm2d(num_features=out_channels))

        if relu is not None:
            layers.append(nn.ReLU() if relu == 0.0 else nn.LeakyReLU(relu))

        self.cbr2d = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr2d(x)