# 声明体系结构
from torch import nn #导入对应库
from tqdm import tqdm #导入进度条
import torch.nn.functional as F #一个包含卷积函数的库
from torchvision import datasets, models
import torch

from attention import model1
import paddle
import numpy as np
import numpy as np
from torchvision import datasets, transforms
# 声明体系结构
class StegNet(nn.Module):  # 定义一个类torch.nn.Module是所有神经网络模块的基类，所有的神经网络模型都应该继承这个基类
    def __init__(self):  # 初始化函数
        super(StegNet, self).__init__()

        self.define_decoder()


    def define_decoder(self):
        self.decoder_layers1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.decoder_layers2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        #         self.decoder_bn2 = nn.BatchNorm2d(256)

        self.decoder_layers3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_layers4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #         self.decoder_bn4 = nn.BatchNorm2d(256)

        self.decoder_layers5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # payload_decoder
        self.decoder_payload1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_payload2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.decoder_payload3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_payload4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.decoder_payload5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_payload6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

        # source_decoder
        self.decoder_source1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_source2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.decoder_source3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_source4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.decoder_source5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_source6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 定义前向传播
        source, payload = x
        # 特殊用法：参数-1 (自动调整size)view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
        s = source.contiguous().view((-1, 3, 256,256 ))

        p = payload.contiguous().view((-1, 3, 256,256))
        encoder_output = s

        # -------------------- Decoder --------------------------

        d = encoder_output.view(-1, 3, 256, 256)

        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))
        #         d = self.decoder_bn2(d)

        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))
        #         d = self.decoder_bn4(d)

        init_d = F.relu(self.decoder_layers5(d))

        # ---------------- decoder_payload ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)

        # ---------------- decoder_source ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_source1(init_d))
        d = F.relu(self.decoder_source2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_source3(d))
        d = F.relu(self.decoder_source4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_source5(d))
        decoded_source = self.decoder_source6(d)

        return  decoded_payload, decoded_source


model = StegNet()
model