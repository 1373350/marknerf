import torch
import lpips
from PIL import Image
import numpy as np

# 假设您已经有了要计算LPIPS距离的两张图片 image1 和 image2
# 加载图像文件
image1 = Image.open("jiema340.png")
image2 = Image.open("2.png")
image1 =image1.convert('RGB')
image2 =image2.convert('RGB')
# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex")

# 将图像转换为PyTorch的Tensor格式
image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# 使用LPIPS模型计算距离
distance = lpips_model(image1_tensor, image2_tensor)

print("LPIPS distance:", distance.item())

