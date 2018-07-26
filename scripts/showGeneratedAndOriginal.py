import cv2
import numpy as np
import os


def mergeImage(im1, im2):
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 != h2 or c1 != c2:
        return im1
    merged = np.zeros((h1, w1+w2, c1))
    merged[:,:w1, :] = im1
    merged[:, w1:, :] = im2
    return merged


def mergeFilesInFolder(folder1, folder2, folder_out):
    folder_images1 = [f for f in os.listdir(folder1)]
    folder_images2 = [f for f in os.listdir(folder2)]
    for i in range(len(folder_images1)):
        im1 = cv2.imread(os.path.join(folder1, folder_images1[i]))
        im2 = cv2.imread(os.path.join(folder2, folder_images2[i]))
        merged_img = mergeImage(im1, im2)
        cv2.imwrite(os.path.join(folder_out, folder_images1[i]), merged_img)


if __name__ == "__main__":
    folder1 = r"C:\Users\Michal\Documents\CycleGAN-tensorflow\datasets\profile2manga_small\testA"
    folder2 = r"C:\Users\Michal\Documents\CycleGAN-tensorflow\test"
    folder_out = r"C:\Users\Michal\Documents\CycleGAN-tensorflow\test_with_original"
    mergeFilesInFolder(folder1, folder2, folder_out)
