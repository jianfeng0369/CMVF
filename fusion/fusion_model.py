import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(in_channels, affine=True)
        self.conv1_relu = nn.ReLU()

        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

        self.conv_res = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)
        self.out_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        residual = self.conv_res(x)

        out = out + residual
        out = self.out_relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.residual_1 = ResidualBlock(in_channels=16, out_channels=16)
        self.residual_2 = ResidualBlock(in_channels=32, out_channels=16)
        self.residual_3 = ResidualBlock(in_channels=48, out_channels=16)

    def forward(self, x):
        out1 = self.residual_1(x)
        out2 = self.residual_2(torch.cat([x, out1], dim=1))
        out3 = self.residual_3(torch.cat([x, out1, out2], dim=1))
        return torch.cat([x, out1, out2, out3], dim=1)

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.conv_in_ir_tm1 =  ConvLayer(1, 2, 1, 1)
        self.conv_in_vis_tm1 = ConvLayer(1, 2, 1, 1)

        self.conv_in_ir_t = ConvLayer(1, 14, 1, 1)
        self.conv_in_vis_t = ConvLayer(1, 14, 1, 1)

        self.dense_ir = DenseBlock()
        self.dense_vis = DenseBlock()

        self.conv_out_1 = ConvLayer(128, 64, 3, 1)
        self.conv_out_2 = ConvLayer(64, 32, 3, 1)
        self.conv_out_3 = ConvLayer(32, 16, 3, 1)
        self.conv_out_4 = ConvLayer(16, 1, 1, 1, is_last=True)

    def forward(self, vi, ir):

        x_in_ir_tm1 = self.conv_in_ir_tm1(ir[:,0:1])
        x_in_vis_tm1 = self.conv_in_vis_tm1(vi[:,0:1])

        x_in_ir_t = self.conv_in_ir_t(ir[:,1:2])
        x_in_vis_t = self.conv_in_vis_t(vi[:,1:2])

        x_in_ir = torch.cat([x_in_ir_tm1, x_in_ir_t], dim=1)
        x_in_vis = torch.cat([x_in_vis_tm1, x_in_vis_t], dim=1)

        x_ir = self.dense_ir(x_in_ir)
        x_vis = self.dense_vis(x_in_vis)

        x_out = torch.cat([x_ir, x_vis], dim=1)

        x_out = self.conv_out_1(x_out)
        x_out = self.conv_out_2(x_out)
        x_out = self.conv_out_3(x_out)
        x_out = self.conv_out_4(x_out)

        return x_out

def get_fusion_model():
    return FusionNet()

