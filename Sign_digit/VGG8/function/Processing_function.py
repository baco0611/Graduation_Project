import cv2
import numpy as np

def resize_list_image(list):
    new_list = []

    for img in list:
        resized_img = cv2.resize(img, (224, 224))
        new_list.append(resized_img)

    new_list = np.array(new_list).reshape(-1, 224, 224, 3)
    return new_list

