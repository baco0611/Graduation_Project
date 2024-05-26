import joblib
import cv2
import os
from random import randint
import numpy as np


def check_folder(destination_folder):
    subfolders = ['raw', 'flip', 'negative', 'resize', 'rotate']
    for subfolder in subfolders:
        subfolder_path = os.path.join(destination_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")

def check_and_create_subfolders(folder, folder_index):
    subfolders = ['raw', 'flip', 'negative', 'resize', 'rotate']
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder, subfolder)
        folder_index_path = os.path.join(subfolder_path, f"{folder_index}")
        if not os.path.exists(folder_index_path):
            os.makedirs(folder_index_path)
            print(f"Tạo thư mục {folder_index_path}")


def process_data(source_folder, destination_folder):
    print(f"\n\nWorking on folder {destination_folder}")
    check_folder(destination_folder)

    print("Processing image data ...")

    folder_index = 0
    raw_list = []
    flip_list = []
    negative_list = []
    resize_list = []
    rotate_list = []
    label = []

    label_name = []
    dog_name = []

    for root, dirs, _ in os.walk(source_folder):
        for directory in dirs:
            # if folder_index == 21:
            #     break
            print(f"\nLoading on folder {directory}, {folder_index} ...")
            
            check_and_create_subfolders(destination_folder, folder_index)

            label_name.append(folder_index)
            dog_name.append(directory.split("-")[-1])

            for _, _, files in os.walk(os.path.join(source_folder, directory)):
                file_index = 1
                for file in files:
                    image_path = os.path.join(source_folder, directory, file)
                    image = cv2.imread(image_path)

                    #　Save to raw folder
                    raw_folder = os.path.join(destination_folder, "raw", str(folder_index))
                    raw_image_path = os.path.join(raw_folder, f"{file_index}.jpg")
                    cv2.imwrite(raw_image_path, image)
                    raw_list.append(image)

                    # Save to flip folder
                    flip_folder = os.path.join(destination_folder, "flip", str(folder_index))
                    flipped_image = cv2.flip(image, 1)
                    flip_image_path = os.path.join(flip_folder, f"{file_index}.jpg")
                    cv2.imwrite(flip_image_path, flipped_image)
                    flip_list.append(flipped_image)

                    # Save to negative folder
                    negative_folder = os.path.join(destination_folder, "negative", str(folder_index))
                    negative_image = cv2.bitwise_not(image)
                    negative_image_path = os.path.join(negative_folder, f"{file_index}.jpg")
                    cv2.imwrite(negative_image_path, negative_image)
                    negative_list.append(negative_image)

                    # Save to resize folder
                    resize_folder = os.path.join(destination_folder, "resize", str(folder_index))
                    scale_factor_x = np.random.uniform(0.5, 1.5)
                    scale_factor_y = np.random.uniform(0.5, 1.5)
                    resized_image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y)
                    resize_image_path = os.path.join(resize_folder, f"{file_index}.jpg")
                    cv2.imwrite(resize_image_path, resized_image)
                    resize_list.append(resized_image)

                    # # Save to rotate folder
                    rotate_folder = os.path.join(destination_folder, "rotate", str(folder_index))
                    angle = randint(-180, 180)
                    rows, cols = image.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
                    rotate_image_path = os.path.join(rotate_folder, f"{file_index}.jpg")
                    cv2.imwrite(rotate_image_path, rotated_image)
                    rotate_list.append(rotated_image)
                    

                    label.append(folder_index)
                    file_index += 1

            folder_index += 1

    process = raw_list + negative_list + resize_list + rotate_list
    print(len(label))
    print(len(raw_list))
    print(len(negative_list))
    print(len(resize_list))
    print(len(rotate_list))
    print(len(flip_list))
    print(len(process))
    # Save processed image lists
    joblib.dump(raw_list, os.path.join(destination_folder, "data", "raw_image.joblib"))
    joblib.dump(flip_list, os.path.join(destination_folder, "data", "flipped_image.joblib"))
    joblib.dump(negative_list, os.path.join(destination_folder, "data", "negative_image.joblib"))
    joblib.dump(resize_list, os.path.join(destination_folder, "data", "resized_image.joblib"))
    joblib.dump(rotate_list, os.path.join(destination_folder, "data", "rotated_image.joblib"))
    joblib.dump(process, os.path.join(destination_folder, "data", "process_image.joblib"))
    joblib.dump(label, os.path.join(destination_folder, "data", "label.joblib"))


# Đường dẫn của thư mục nguồn và thư mục đích
source_folder = "../dataset/images"
destination_folder = "../dataset"

# Xử lý dữ liệu
process_data(source_folder, destination_folder)
