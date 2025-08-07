import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=8):
        super().__init__()
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64,128)
        self.d3 = DoubleConv(128,256)
        self.d4 = DoubleConv(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(512,512,2,stride=2)
        self.u3 = DoubleConv(512+256,256)
        self.u2 = DoubleConv(256+128,128)
        self.u1 = DoubleConv(128+64,64)
        self.out = nn.Conv2d(64, out_classes,1)

    def forward(self,x):
        c1 = self.d1(x); p1=self.pool(c1)
        c2 = self.d2(p1); p2=self.pool(c2)
        c3 = self.d3(p2); p3=self.pool(c3)
        c4 = self.d4(p3)
        u = self.up(c4)
        u = self.u3(torch.cat([u,c3],1))
        u = F.interpolate(u,scale_factor=2,mode='bilinear',align_corners=True)
        u = self.u2(torch.cat([u,c2],1))
        u = F.interpolate(u,scale_factor=2,mode='bilinear',align_corners=True)
        u = self.u1(torch.cat([u,c1],1))
        return self.out(u)
