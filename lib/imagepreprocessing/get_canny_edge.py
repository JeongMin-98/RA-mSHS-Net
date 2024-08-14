# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 입력 이미지 로드
image_path = 'output_image_mean_kuwahara.png'
gau_mean = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_path_2 = 'output_image_gaussian_kuwahara.png'
gau_ka = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

# # CLAHE를 사용하여 히스토그램 평활화를 적용합니다
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized_image = clahe.apply(image)

# # 가우시안 블러를 적용하여 노이즈를 줄입니다
# blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 1.4)

# Canny edge 검출을 수행합니다
edges_ka_gau = cv2.Canny(gau_ka, 50, 150)
edges_ka_mean = cv2.Canny(gau_mean, 50, 150)

# 결과를 표시합니다
plt.figure(figsize=(10, 6))

plt.subplot(1, 4, 1)
plt.title('Original gau Image')
plt.imshow(gau_ka, cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Original mean Image')
plt.imshow(gau_mean, cmap='gray')

plt.subplot(1, 4, 3)
plt.title('gau_ka Image')
plt.imshow(edges_ka_gau, cmap='gray')

plt.subplot(1, 4, 4)
plt.title('mean_ka Edges')
plt.imshow(edges_ka_mean, cmap='gray')

plt.show()

print("Finish")