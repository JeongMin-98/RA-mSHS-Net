import cv2
import numpy as np
from pykuwahara import kuwahara


def get_canny_edge(image):
    return cv2.Canny(image, 50, 150)


def get_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 1.4)


def histogram_equalization(img):
    """ 
        result => equalization Image
    """
    # 1
    (N, M) = img.shape

    G = 256  # gray levels
    H = np.zeros(G)  # initialize an array Histogram

    # 2    
    for g in img.ravel():
        H[g] += 1

    g_min = np.min(np.nonzero(H))

    # 3
    H_c = np.zeros_like(H)  # cumulative image histogram
    H_c[0] = H[0]
    for g in range(1, G):
        H_c[g] = H_c[g - 1] + H[g]

    H_min = H_c[g_min]

    # 4
    T = np.round((H_c - H_min) / (M * N - H_min) * (G - 1))

    # 5
    result = np.zeros_like(img)
    for n in range(N):
        for m in range(M):
            result[n, m] = T[img[n, m]]

    return result


def get_kuwahara(img):
    return kuwahara(img, method='gaussian', radius=3)


def get_mean_filter(img):
    return cv2.blur(img, (5, 5))


def get_median_blur(img):
    return cv2.medianBlur(img, 5)
