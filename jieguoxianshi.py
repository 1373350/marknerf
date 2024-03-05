import matplotlib.pyplot as plt
import torch
import cv2
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from coder import model
from train import s,p
with torch.no_grad():
    model.eval()
    dp,ds= model.forward((s, p))
dp,ds = dp.cpu(), ds.cpu()
ss, pp = s.cpu(), p.cpu()
i=0
#eimage = e.contiguous().view((-1, 256, 256, 3)).numpy()eimage=eimage[i]
dpimage = dp.contiguous().view((-1, 256, 256, 3)).numpy()
dpimage=dpimage[i]

sourceimage, payloadimage = ss.contiguous().view((-1, 256, 256, 3)).numpy(), pp.contiguous().view((-1, 256, 256, 3)).numpy()
sourceimage=sourceimage[i]
payloadimage=payloadimage[i]
plt.figure(figsize=(8, 8))  # 绘图尺寸
plt.subplot(1, 1, 1)  # 画布分割2行3列取第一块
plt.imshow(cv2.cvtColor(dpimage, cv2.COLOR_BGR2RGB))
plt.axis('off')
figure_save_path = "test"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path , 'jiema.png'))
#plt.imshow()  # 显示内容编码结果混合图像
"""plt.subplot(1, 4, 2)
plt.title('Decoded Payload')  # 解码结果秘密图像
plt.imshow(cv2.cvtColor(dpimage, cv2.COLOR_BGR2RGB))
#plt.imshow(dpimage, cmap='gray')
plt.subplot(1, 4, 3)

plt.title('Original Source')  # 源图像
plt.imshow(cv2.cvtColor(sourceimage, cv2.COLOR_BGR2RGB))
#plt.imshow(sourceimage)
plt.subplot(1, 4, 4)
plt.title('Original Payload')  # 秘密图像
plt.imshow(cv2.cvtColor(payloadimage, cv2.COLOR_BGR2RGB))
#plt.imshow(payloadimage, cmap='gray')"""

plt.show()  # plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来。