# 计算秘密图片的归一化的均值和方差
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils,datasets
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
img_h2, img_w2= 256, 256
payload_data=cv.imread('shuiyin.png')
payload_data = cv.resize(payload_data, (img_w2, img_h2))
#newArr2 = [] #创建一个空的列表
#for i2 in (payload_data):   #读取数据集中的图像数据和标签
        #i2=i2.permute(1,2,0)  #将3*256*256转换为256*256*3便于图像显示
        #v2=i2.numpy()         #将tensor转化为numpy
        #newArr2.append(v2)    #把numpy一个个存入到空列表中
        #payload_train= np.array(newArr2)#把list转化为array便于图像显示

#payload_train = np.array([np.array(Image.fromarray(np.uint8(np.interp(img, (img.min(), img.max()), (0, 255)))).convert('L')) for img in payload_train])
payload_data=payload_data/255
payload_train = torch.from_numpy(payload_data).to(device).float()