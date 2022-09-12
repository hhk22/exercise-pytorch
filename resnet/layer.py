import os
import numpy as np

import torch
import torch.nn as nn

class CBR2d(nn.Module):
    def __init__(self, in_channls, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channls,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
        )
        
        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        elif norm == 'inorm':
            layers.append(nn.InstanceNorm2d(num_features=out_channels))
        
        if relu == 0.0:
            layers.append(nn.ReLU())
        elif relu and relu > 0.0:
            layers.append(nn.LeakyReLU(relu))
        
        self.cbr = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers.append(
            CBR2d(
                in_channls=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                norm=norm,
                relu=relu
            )
        )

        layers.append(
            CBR2d(
                in_channls=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                norm=norm,
                relu=None
            )
        )

        self.res_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.res_block(x)


class PixelShuffle(nn.Module):
    def __init__(self, ry, rx):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x

        


