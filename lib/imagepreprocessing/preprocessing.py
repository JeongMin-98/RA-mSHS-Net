from PIL import Image
import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import tools


class palette:
    def __init__(self, image_path, methods: list):
        # root = "./"
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.methods = methods
        self.after_image = []
        self.last_apply_method = []

    def apply_method(self):

        temp = copy.deepcopy(self.image)
        for m in self.methods:
            temp = eval('tools.' + m)(temp)
            self.after_image.append(temp)
            self.last_apply_method.append(m)
        return

    def plot_image(self):
        h = 1
        w = len(self.methods) // h
        plt.figure(figsize=(10, 6))

        for i in range(1, len(self.methods) + 1):
            plt.subplot(h, w, i)
            plt.title(f"{self.last_apply_method[i - 1]}")
            plt.imshow(self.after_image[i - 1], cmap='gray')

        plt.show()

    def save_image(self):
        cv2.imwrite(filename='./output.png', img=self.after_image[-1])


if __name__ == '__main__':
    image_path = 'test/1.png-roi-0.png'
    p = palette(image_path=image_path,
                methods=['histogram_equalization', 'get_gaussian_blur', 'get_canny_edge'])
    p.apply_method()
    p.plot_image()
    p.save_image()
