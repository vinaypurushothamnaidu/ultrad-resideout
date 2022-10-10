import torch.nn.functional as F
import torch
import torch.nn as nn 
from unet_parts import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        factor = 2 if bilinear else 1
        self.down6 = Down(2048, 4096 // factor)
        self.up1 = Up(4096, 2048 // factor, bilinear)
        self.up2 = Up(2048, 1024 // factor, bilinear)
        self.up3 = Up(1024, 512 // factor, bilinear)
        self.up4 = Up(512, 256 // factor, bilinear)
        self.up5 = Up(256, 128 // factor, bilinear)
        self.up6 = Up(128, 64, bilinear)
        self.pre = nn.Conv2d(64, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.re(self.pre(x))
        return x
