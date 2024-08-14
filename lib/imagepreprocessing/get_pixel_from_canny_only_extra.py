# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 로드합니다
# image_path = 'test/1.png-roi-0.png'
image_path = 'output_image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Canny edge 검출을 수행합니다
edges = cv2.Canny(image, 50, 150)

# 최 외각 테두리를 검출합니다
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 최 외각 테두리만 포함한 새 이미지를 만듭니다
external_contours = np.zeros_like(image)
cv2.drawContours(external_contours, contours, -1, 255, 1)

# 테두리 간의 거리를 계산합니다
def calculate_min_distance(contours):
    min_distance = float('inf')
    for i in range(len(contours)):
        for j in range(i+1, len(contours)):
            for point1 in contours[i]:
                for point2 in contours[j]:
                    distance = np.linalg.norm(point1 - point2)
                    if distance < min_distance:
                        min_distance = distance
    return min_distance

# 외곽선 간의 최소 거리를 계산합니다
min_distance = calculate_min_distance(contours)

# 결과를 출력합니다
print(f"최 외각 테두리 간의 최소 픽셀 거리: {min_distance}")

# 결과를 표시합니다
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('External Contours')
plt.imshow(external_contours, cmap='gray')

plt.show()
