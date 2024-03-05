"""import math
import numpy as np
import cv2
def psnr(target, ref):
  target_data = np.array(target, dtype=np.float64)
  ref_data = np.array(ref,dtype=np.float64)
  # 直接相减，求差值
  diff = ref_data - target_data
  # 按第三个通道顺序把三维矩阵拉平
  diff = diff.flatten('C')
  # 计算MSE值
  rmse = math.sqrt(np.mean(diff ** 2.))
  # 精度
  eps = np.finfo(np.float64).eps
  if(rmse == 0):
    rmse = eps
  return 20*math.log10(255.0/rmse)
def ssim(imageA, imageB):
  # 为确保图像能被转为灰度图
  imageA = np.array(imageA, dtype=np.uint8)
  imageB = np.array(imageB, dtype=np.uint8)
  #  通道分离，注意顺序BGR不是RGB
  (B1, G1, R1) = cv2.split(imageA)
  (B2, G2, R2) = cv2.split(imageB)
  # convert the images to grayscale BGR2GRAY
  grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
  (score0, diffB) = compare_ssim(B1, B2, full=True)
  (score1, diffG) = compare_ssim(G1, G2, full=True)
  (score2, diffR) = compare_ssim(R1, R2, full=True)
  aveScore = (score0+score1+score2)/3
  print("BGR average SSIM: {}".format(aveScore ))
  """