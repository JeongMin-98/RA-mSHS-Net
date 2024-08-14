import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 이미지 읽기
image_path = 'test/1.png-roi-0.png'
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# 이미지 크기와 중심 계산
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# 마스크 생성 (원형)
mask = np.ones((rows, cols, 2), np.uint8)
r = 80  # 원형 마스크의 반경
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]

# 원형 마스크 생성
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r ** 2
mask[mask_area] = 1  # 원형 영역 안의 주파수 성분을 제거


# 푸리에 변환 수행
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 주파수 스펙트럼 계산
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# 마스크 적용 및 필터링된 스펙트럼 계산
fshift = dft_shift * mask
fshift_magnitude = cv.magnitude(fshift[:, :, 0], fshift[:, :, 1])
fshift_magnitude += 1e-8  # 작은 값 추가
fshift_mask_mag = 2000 * np.log(fshift_magnitude)

# 역 푸리에 변환을 통해 이미지 복원
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# 노이즈 제거 및 부드럽게 하기 위해 Gaussian 블러링 적용
img_back_blur = cv.GaussianBlur(img_back, (3, 3), 0)

# Canny edge detection
edges = cv.Canny(img_back_blur.astype(np.uint8), 100, 150)

# 결과 출력
plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(img_back_blur, cmap='gray')
plt.title('After Gaussian Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(edges, cmap='gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
plt.show()
