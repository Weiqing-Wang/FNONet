import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # 初始化可学习的复数权重
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # 复数乘法
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # (32, 64, 172, 79)
        batchsize = x.shape[0]
        # 进行二维快速傅里叶变换
        x_ft = torch.fft.rfft2(x, s=(x.size(-2), x.size(-1))) #(32, 64, 172, 40)

        # 初始化输出的傅里叶系数
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        # 应用可学习的权重
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # 进行二维逆快速傅里叶变换
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNONet(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNONet, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # 输入层
        self.fc0 = nn.Conv2d(in_channels=5, out_channels=self.width,kernel_size=1)

        # 谱卷积层
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # 输出层
        self.fc1 = nn.Conv2d(self.width, 64,kernel_size=1)
        self.fc2 = nn.Conv2d(64, 3,kernel_size=1)
    def get_grid(self, shape, device):

        batchsize, h, w = shape[0], shape[-2], shape[-1]


        gridx = torch.tensor(np.linspace(0, 2.6, w), dtype=torch.float)

        gridx = gridx.reshape(1, 1, w).repeat([batchsize, h, 1])


        gridy = torch.tensor(np.linspace(0, 1.2, h), dtype=torch.float)

        gridy = gridy.reshape(1, h, 1).repeat([batchsize, 1, w])


        grid = torch.stack([gridx, gridy], dim=1)

        return grid.to(device)
    def forward(self, x):
        grids=self.get_grid(x.shape,x.device)
        x=torch.cat((x,grids),dim=1)
        x=self.fc0(x)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        #
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

