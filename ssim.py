import cv2
from skimage import metrics

# Assume you have two image files: image1.jpg and image2.jpg

# Load the images
image1 = cv2.imread("daxiang.png")
image2 = cv2.imread("boshong.png")

image1 = cv2.resize(image1, (256, 256))
image2 = cv2.resize(image2, (256, 256))
# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute SSIM
ssim_score = metrics.structural_similarity(gray_image1, gray_image2)

print("SSIM score:", ssim_score)
