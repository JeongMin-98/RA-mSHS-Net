import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 로드합니다
image_path = 'test/1.png-roi-0.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Sobel 필터를 사용하여 y 방향의 경계를 검출합니다
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 절대값을 취하여 음수 값을 양수로 변환합니다
sobel_y = np.absolute(sobel_y)

# 결과를 8비트로 변환합니다
sobel_y = np.uint8(sobel_y)

# 이진화 과정을 추가합니다
_, binary_sobel_y = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)

# 컨투어를 검출합니다
contours, _ = cv2.findContours(binary_sobel_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 검출된 컨투어를 원본 이미지에 그립니다
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# 결과를 표시합니다
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 4, 2)
plt.title('Sobel Y (Vertical Edges)')
plt.imshow(sobel_y, cmap='gray')

plt.subplot(1, 4, 3)
plt.title('Binary Sobel Y')
plt.imshow(binary_sobel_y, cmap='gray')

plt.subplot(1, 4, 4)
plt.title('Contours')
plt.imshow(contour_image)

plt.show()
