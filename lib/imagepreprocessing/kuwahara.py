import cv2
import numpy as np
import matplotlib.pyplot as plt
from pykuwahara import kuwahara


# 이미지 불러오기
src = cv2.imread('test/1.png-roi-0.png')

# 필터 크기 설정
ksize = 5

# Kuwahara 필터 적용
dst = kuwahara(src, method='mean', radius=3)
dst2 = kuwahara(src, method='gaussian', radius=3)

plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(src, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('gaussian kuwahara Image')
plt.imshow(dst, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Kuwahara')
plt.imshow(dst, cmap='gray')

plt.show()

# # 결과 이미지 저장
cv2.imwrite('output_image_mean_kuwahara.png', dst)
cv2.imwrite('output_image_gaussian_kuwahara.png', dst2)
