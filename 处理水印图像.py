import cv2
import os
import matplotlib.pyplot as plt
img = cv2.imread("shuiyin.png")

image_size = [256, 256]

img = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
plt.figure(figsize=(8, 8))  # 绘图尺寸
plt.subplot(1, 1, 1)  # 画布分割2行3列取第一块
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
figure_save_path = "test"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path , '1.png'))
plt.show()
print("imag.shap: {}".format(img.shape))