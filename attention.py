from turtle import forward
import torch
from torch import nn


# channel注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )

        self.sigmoid = nn.Sigmoid()
        # 以上为网络的定义部分

    def forward(self, x):  # 前向传播，把网络串起来
        b, c, h, w = x.size()
        out_max_pool = self.max_pool(x).view([b, c])
        out_avg_pool = self.avg_pool(x).view([b, c])
        out_fc_max = self.fc(out_max_pool)
        out_fc_avg = self.fc(out_avg_pool)
        out = out_fc_avg + out_fc_max
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


# spatial注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        padding = 3 // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_avg = torch.mean(x, dim=1, keepdim=True)
        out_max, _ = torch.max(x, dim=1, keepdim=True)
        out_pool = torch.cat([out_max, out_avg], dim=1)
        out = self.conv1(out_pool)
        out = self.sigmoid(out)

        return out * x


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        # 定义部分
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out_channel_attention = self.channel_attention(x)
        out_spatial_attention = self.spatial_attention(out_channel_attention)
        return out_spatial_attention


model1 = CBAM(3)
