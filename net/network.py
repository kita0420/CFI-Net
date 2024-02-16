import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torchvision.ops import DeformConv2d
from networks.othernets.Modules import *
from torchvision.transforms.functional import resize
import cv2
from networks.othernets.mlp import *
from method import *
class CFI_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(CFI_Net, self).__init__()

        self.chan = [32, 64, 128, 256, 512]
        # self.chan = [64, 128, 256, 512, 1024]
        # self.chan = [64, 96, 128, 256, 512]
        chan = self.chan

        self.P1 = JRDM(C = self.chan[0],size = 2)
        self.P2 = JRDM(C=self.chan[1], size=2)
        self.P3 = JRDM(C=self.chan[2], size=2)
        self.P4 = JRDM(C=self.chan[3], size=2)

        self.inner_attn1 = CIAM(C = self.chan[0],size = 960//2)
        self.inner_attn2 = CIAM(C = self.chan[1],size = 480//2)
        self.inner_attn3 = CIAM(C = self.chan[2],size = 240//2)
        self.inner_attn4 = CIAM(C = self.chan[3],size = 120//2)
        # self.inner_attn_out = Inner_attn(C = 64,size = 16)

        # self.P1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.P2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        # self.P3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        # self.P4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        self.Conv1 = conv_block0(img_ch, self.chan[0])
        self.Conv2 = conv_block1(self.chan[0], self.chan[1])
        self.Conv3 = conv_block1(self.chan[1], self.chan[2])
        self.Conv4 = conv_block1(self.chan[2], self.chan[3])
        self.Conv5 = conv_block1(self.chan[3], self.chan[4])


        self.Up5 = self.upconv(self.chan[4], self.chan[3])
        self.Up_conv5 = conv_block1(self.chan[3]*2, self.chan[3])

        self.Up4 = self.upconv(self.chan[3], self.chan[2])
        self.Up_conv4 = conv_block1(self.chan[2]*2, self.chan[2])

        self.Up3 = self.upconv(self.chan[2], self.chan[1])
        self.Up_conv3 = conv_block1(self.chan[1]*2, self.chan[1])

        self.Up2 = self.upconv(self.chan[1], self.chan[0])
        self.Up_conv2 = conv_block1(self.chan[0]*2, self.chan[0])

        self.Conv_1x1 = nn.Conv2d(self.chan[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # self.affr1 = ASCGM(in_ch=1, size=3)

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.P1(x1)
        x2 = self.Conv2(x2)

        x3 = self.P2(x2)
        x3 = self.Conv3(x3)

        x4 = self.P3(x3)
        x4 = self.Conv4(x4)

        x5 = self.P4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.inner_attn4(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.inner_attn3(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.inner_attn2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.inner_attn1(d2)

        d1 = self.Conv_1x1(d2)
        # d1 = self.affr1(d1)
        d1 = self.sigmoid(d1)


        return d1