import os
import numpy as np

import torch
import torch.nn as nn

from layer import CBR2d, ResBlock, PixelShuffle

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm='bnorm', nblk=16):
        super().__init__()
        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)

        self.fc = CBR2d(nker, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)
    
    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x

class SRResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm='bnorm', nblk=16):
        super().__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(
            in_channls=in_channels,
            out_channels=nker,
            kernel_size=9,
            padding=4,
            stride=1,
            norm=None,
            bias=True,
            relu=0.0
        )

        res = []
        for _ in range(nblk):
            res.append(
                ResBlock(
                        in_channels=nker,
                        out_channels=nker,
                        kernel_size=3, 
                        stride=1,
                        padding=1,
                        bias=True,
                        norm=norm,
                        relu=0.0
                )
            )
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(
            in_channls=nker,
            out_channels=nker,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            norm=norm,
            relu=None
        )

        ps1 = []
        ps1.append(
            nn.Conv2d(
                in_channels=nker,
                out_channels=4*nker,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True
            )
        )
        ps1.append(PixelShuffle(ry=2, rx=2))
        ps1.append(nn.AvgPool2d(kernel_size=2))
        ps1.append(nn.ReLU())
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2.append(
            nn.Conv2d(
                in_channels=nker,
                out_channels=4*nker,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True
            )
        )
        ps2.append(PixelShuffle(ry=2, rx=2))
        ps2.append(nn.AvgPool2d(kernel_size=2))
        ps2.append(nn.ReLU())
        self.ps2 = nn.Sequential(*ps2)

        self.fc = nn.Conv2d(
            in_channels=nker,
            out_channels=out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=True
        )
    
    def forward(self, x):
        x = self.enc(x)
        x0 = x
        
        x = self.res(x)
        x = self.dec(x)
        x = x + x0

        x = self.ps1(x)
        x = self.ps2(x)
        
        if self.learning_type == 'plain':
            x = self.fc(x)
        elif self.learning_type == 'residual':
            x = self.fc(x) + x0

        return x

