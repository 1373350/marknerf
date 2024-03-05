from coder import model
from attention import model1
from torch import nn
import matplotlib.pyplot as plt
import torch
import cv2
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from payloadimage import payload_train
from sourceimage import source_train
from coder import model
model.cuda()
path="state_dict_model.pt"
model.load_state_dict(torch.load(path))
model.eval()
s, p = source_train, payload_train
# s为source_train中第0个元素到第64个元素的数据，p为 payload_train中第0个元素到第64个元素的数据
s.to(device)
p.to(device)
with torch.no_grad():
    model.eval()
dp, ds = model.forward((s, p))
dp,ds = dp.cpu(), ds.cpu()
i=0
dpimage = dp.contiguous().view((-1, 256, 256, 3)).detach().numpy()
dpimage=dpimage[i]
plt.figure(figsize=(8, 8))  # 绘图尺寸
plt.subplot(1, 1, 1)  # 画布分割2行3列取第一块
plt.imshow(cv2.cvtColor(dpimage, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()