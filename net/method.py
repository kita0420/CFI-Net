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

class JRDM(nn.Module):
    def __init__(self,C,size): #C->channel size->patch size
        super(JRDM, self).__init__()
        self.ker_size = size
        self.unfold1 = nn.Unfold(kernel_size=(self.ker_size,self.ker_size),stride = (self.ker_size,self.ker_size))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=C*size*size,out_channels=C,kernel_size=1,stride=1,padding=0),
                                   # nn.BatchNorm2d(num_features=C),
                                   # nn.ReLU(inplace=True),
                                   )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=C * 2, out_channels=C, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=C),
            nn.ReLU(inplace=True),
            )
        self.Maxpool =  nn.Conv2d(C, C, 3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.Maxpool(x)
        size = self.ker_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        [B, C, H, W] = x.shape
        # x = x.reshape(B, -1, C, H, W).contiguous()
        xu1 = self.unfold1(x)
        xu1 = xu1.reshape(B, C*4, H//size,W//size)
        xu1 = self.conv1(xu1)
        xu1 = torch.cat((xu1,x1),dim = 1)
        xu1 = self.conv2(xu1)
        return xu1


class channel_attention1(nn.Module):
    def __init__(self, in_channels):
        super(channel_attention1,self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=1),
        #     nn.BatchNorm2d(in_channels//2),
        #     # nn.ReLU(inplace=True),
        #     Swish(),
        #     nn.Conv2d(in_channels//2, 1, kernel_size=1, stride=1),
        #     # nn.BatchNorm2d(1),
        #     # Swish(),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_channels),
            # Swish(),

        )
        self.sig = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #     # nn.BatchNorm2d(in_channels),
        #     # nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(in_channels),
        #     Swish(),
        # )
        #self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, y):
        MaxP = nn.MaxPool2d(kernel_size=y.shape[2],stride = y.shape[2])
        x2 = MaxP(y)
        x2 = self.conv(x2)
        x2 = self.sig(x2)
        out = y * x2
        # out = self.conv1(out)

        # x1 = torch.max(out, dim=1, keepdim=True).values
        # x1 = self.sig(x1)
        # attn = x1
        # out = out * attn
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CIAM(nn.Module): #C->channel  size->patch size K
    def __init__(self,C,size):
        super(CIAM, self).__init__()
        self.ker_size = size
        self.fc = nn.Linear(in_features=C, out_features=C)
        self.unfold1 = nn.Unfold(kernel_size=(self.ker_size,self.ker_size),stride = (self.ker_size,self.ker_size))
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0)
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv1.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()
        self.LN = nn.LayerNorm(C)

    def forward(self, x):
        size = self.ker_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        [B, C, H, W] = x.shape
        # x = x.reshape(B, -1, C, H, W).contiguous()
        xu1 = self.unfold1(x)
        xu1 = xu1.permute(0, 2, 1).view(B, -1, C, size*size) # b c h w -> b h*w/k*k c k*k



        mean = torch.max(xu1,dim=3).values
        mean = self.fc(mean)
        mean = self.sigmoid(mean)
        mean = mean.reshape(B,mean.shape[1],-1,1).contiguous() #b
        xu1 = xu1*mean

        xu2max = torch.max(xu1, dim=2, keepdim=True).values
        # # xu2max = self.sigmoid(xu2max)
        xu2mean = torch.mean(xu1, dim=2, keepdim=True)
        xu2 = torch.cat((xu2mean, xu2max), dim=2)
        xu2 = xu2.reshape(-1, 2, size, size)
        xu2 = self.conv(xu2)
        # xu2 = xu2max
        xu2 = self.sigmoid(xu2)
        xu2 = xu2.reshape(B, -1, 1, size * size)
        # # xu1 = xu1 * xu2
        xu1 = xu1 * xu2

        xu1 = xu1.view(B, -1, C*size*size).permute(0, 2, 1).contiguous()
        fold1 = nn.Fold(output_size=(H, W), kernel_size=(size, size), stride=(size, size)).to(device)
        xu1 = fold1(xu1)

        # out = self.Maxpool(xu1)

        xu3 = self.unfold1(xu1)
        xu3 = xu3.reshape(B, C*size*size, H//size,W//size)
        xu3max = torch.max(xu3, dim=1, keepdim=True).values
        xu3mean = torch.mean(xu3, dim=1, keepdim=True)
        xu3 = torch.cat((xu3mean, xu3max), dim=1)
        xu3 = self.conv2(xu3)
        xu3 = self.sigmoid(xu3)
        xu3 = torch.repeat_interleave(xu3, size,dim =3)
        xu3 = torch.repeat_interleave(xu3, size, dim=2)
        xu1 = xu1 * xu3
        return xu1


class ASCGM(nn.Module):
    def __init__(self,in_ch = 64 ,size = 3):  #size =3,5,7    means neighbor size*size windows
        super(ASCGM,self).__init__()
        self.size = size
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.pad3 = nn.ReflectionPad2d(3)
        self.softmax = nn.Softmax(dim = 3)
        self.unfold1 = nn.Unfold(kernel_size=(1,1),stride = (1,1))
        self.unfold3 = nn.Unfold(kernel_size=(3, 3), stride=(1,1))
        self.unfold5 = nn.Unfold(kernel_size=(5, 5), stride=(1, 1))
        self.unfold7 = nn.Unfold(kernel_size=(7, 7), stride=(1, 1))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch , kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_ch),
            # nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch*2, out_channels=in_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, dd):
        d = self.conv1(dd)
        B, C, H, W = d.shape

        d1 = self.unfold1(d)
        d1 = d1.permute(0, 2, 1).view(B, H * W, C, 1 * 1)
        if self.size == 3:
            dd3 = self.pad1(d)
            dd3 = self.unfold3(dd3)
            dd3 = dd3.permute(0, 2, 1).view(B, H * W, C, 3 * 3)  # neighbor 9 points
            affr_d3 = dd3-d1
            affr_d3_abs = torch.abs(dd3 - d1)
            affr3_avg = torch.median(affr_d3_abs, dim=3, keepdim=True).values
            affr_d3[affr_d3_abs > affr3_avg] = 0
            affr_d3_abs[affr_d3_abs > affr3_avg] = 0
            # affr_d3S = 1 - self.softmax(affr_d3)
            affr_d3S = affr_d3_abs / torch.max(affr_d3_abs, dim=3, keepdim=True).values
            d3 = affr_d3 * (1 - affr_d3S)
            d3 = torch.sum(d3, dim=3)
            d3 = d3.permute(0, 2, 1).view(B, C, H, W)  # neighbor 9 points
            d3 = torch.cat((d, d3), dim=1)
            d3 = self.conv2(d3)
            return d3
        elif self.size == 5:
            dd5 = self.pad2(d)
            dd5 = self.unfold5(dd5)
            dd5 = dd5.permute(0, 2, 1).view(B, H * W, C, 5 * 5) # neighbor 25 points
            affr_d5 = d1 * dd5
            affr_d5 = self.softmax(affr_d5)
            d5 = affr_d5 * dd5
            d5 = torch.sum(d5, dim=3)
            d5 = d5.permute(0, 2, 1).view(B, C, H, W)  # neighbor 25 points
            # out = torch.cat((d, d5), dim=1)
            d5 = self.conv2(d5)
            return d5+d
        elif self.size == 7:
            dd7 = self.pad3(d)
            dd7 = self.unfold7(dd7)
            dd7 = dd7.permute(0, 2, 1).view(B, H * W, C, 7 * 7)  # neighbor 49 points
            affr_d7 = d1 * dd7
            affr_d7 = self.softmax(affr_d7)
            d7 = affr_d7 * dd7
            d7 = torch.sum(d7, dim=3)
            d7 = d7.permute(0, 2, 1).view(B, C, H, W)  # neighbor 49 points
            # out = torch.cat((d, d7), dim=1)
            d7 = self.conv2(d7)
            return d7+d
        else:
            return d