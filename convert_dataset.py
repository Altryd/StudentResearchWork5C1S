import cv2
import numpy as np
from scipy.ndimage import median_filter
import os
import matplotlib.pyplot as plt

for filepath in os.scandir("Samara/original"):
    if not filepath.is_file():
        continue
    original_image = cv2.imread(filepath)
    edited_image = original_image.copy()
    indexes_pen = np.where(edited_image[:, :, 0] > 20)
    indexes_bg = np.where(np.logical_and(edited_image[:, :, 1] > 215, edited_image[:, :, 2] > 215))
    # indexes_pen = np.where(np.logical_and(first_image[:, :, 0] > 20, np.logical_and(first_image[:, :, 1] < 215, first_image[:, :, 2] < 215)))
    edited_image[indexes_pen] = 255
    edited_image[indexes_bg] = 0

    edited_image = median_filter(edited_image, 2)

    # cv2.imshow("test", np.concatenate((original_image, edited_image)))
    cv2.imwrite(f"Samara_inverse/original_inverse/{filepath.name}", edited_image)
    # cv2.waitKey(0)

