import cv2
import numpy as np
import glob
import os
import random


def load_images(image_dir):
    import joblib
    images = joblib.load(image_dir)

    bgr_images = []
    gray_images = []

    i = 1

    for bgr in images:
        # print(i, end="\t")
        # i+=1
        # bgr = cv2.imread(image_path)
        # cv2.imshow("Image", bgr)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr_images.append(bgr)
        gray_images.append(gray)

    return bgr_images, gray_images


def extract_visual_features(gray_images):
# Extract SIFT features from gray images
    # Define our feature extractor (SIFT)
    extractor = cv2.SIFT_create()
    
    keypoints = []
    descriptors = []

    for img in gray_images:
        # extract keypoints and descriptors for each image
        img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
        if img_descriptors is not None:
            keypoints.append(img_keypoints)
            descriptors.append(img_descriptors)
    return keypoints, descriptors


def visualize_keypoints(bgr_image, image_keypoints):
    image = bgr_image.copy()
    image = cv2.drawKeypoints(image, image_keypoints, 0, (0, 255, 0), flags=0)
    return image

