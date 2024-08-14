# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 로드하고 그레이스케일로 변환합니다
image = cv2.imread('test/1.png-roi-0.png', cv2.IMREAD_GRAYSCALE)

# CLAHE를 사용하여 히스토그램 평활화를 적용합니다
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_image = clahe.apply(image)

# 가우시안 블러를 적용하여 노이즈를 줄입니다
blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 1.4)

# Canny edge 검출을 수행합니다
edges = cv2.Canny(blurred_image, 50, 150)

# 경계를 검출합니다
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 검출된 경계를 원본 이미지에 그립니다
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

# 특정 관심 영역(ROI)을 설정하여 관절 간의 간격을 측정합니다
# 여기에 관심 영역(ROI)을 설정하는 코드를 추가합니다
# 예를 들어, 특정 좌표를 기준으로 하는 사각형 영역을 설정할 수 있습니다

# 관심 영역을 설정합니다 (예시: 이미지의 중앙 부분)
height, width = image.shape
roi = edges[height//4:3*height//4, width//4:3*width//4]

# 관심 영역 내에서 경계를 다시 검출합니다
roi_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 경계선 사이의 거리를 계산합니다
distances = []
for cnt in roi_contours:
    for i in range(len(cnt) - 1):
        for j in range(i + 1, len(cnt)):
            point1 = cnt[i][0]
            point2 = cnt[j][0]
            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)

print(distances)
# 가장 짧은 거리(즉, 관절 간의 간격)를 찾습니다
if distances:
    min_distance = sum(distances) / len(distances)
else:
    min_distance = None

# 결과를 출력합니다
print(f"관절 간의 최소 간격: {min_distance}")

# 결과를 표시합니다
plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Contours')
plt.imshow(contour_image)

plt.show()
