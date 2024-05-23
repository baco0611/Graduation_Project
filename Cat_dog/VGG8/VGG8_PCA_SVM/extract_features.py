from PCA_SVM_config import *
import joblib
import numpy as np
from sklearn.decomposition import PCA
import os

def concatenate_data(data):
    return np.concatenate(data)

def apply_pca(data, PCA_dims):
    pca = PCA(n_components=PCA_dims)
    pca.fit(data)
    return pca


def train_and_save_PCA(data_dims):
    print(f"\n\n\nTraining for {data_dims}-dims vector")

    for PCA_dims in range(200, 600, 100):
        print(f"\nTraining PCA model with {PCA_dims} dims")
        model_name = f"{date}_PCA_" + str(PCA_dims) + "_" + str(data_dims)

        data_folder = f"{data_dims}dims_data"
        folder = f"../VGG8_SVM/data/dataset/{data_folder}/raw_image_features.joblib"
        folder = f"../VGG8_SVM/data/dataset/{data_folder}/process_image_features.joblib"

        print(f"Loading data ...")
        data = joblib.load(folder)

        print(f"Training model ...")
        model = apply_pca(data, PCA_dims)
        joblib.dump(model, f"./data/model/{model_name}.joblib")

        sub_folder_path = os.path.join("./data/dataset", str(PCA_dims))
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)

        folders = [
            f"../VGG8_SVM/data/dataset/{data_folder}/raw_image_features.joblib",
            f"../VGG8_SVM/data/dataset/{data_folder}/negative_image_features.joblib",
            f"../VGG8_SVM/data/dataset/{data_folder}/resized_image_features.joblib",
            f"../VGG8_SVM/data/dataset/{data_folder}/rotated_image_features.joblib",
            f"../VGG8_SVM/data/dataset/{data_folder}/flipped_image_features.joblib",
        ]

        print(f"Extract feature ...")
        ft = np.empty((0, PCA_dims))
        for folder in folders:
            folder_list = folder.split("/")
            name = folder_list[-1]

            data = joblib.load(folder)

            result = model.transform(data)
            joblib.dump(result, f"./data/dataset/{PCA_dims}/{data_dims}_{name}")

            ft = np.vstack((ft, result))
        folder = f"../VGG8_SVM/data/dataset/{data_folder}/process_image_features.joblib"
        folder_list = folder.split("/")
        name = folder_list[-1]
        joblib.dump(ft, f"./data/dataset/{PCA_dims}/{data_dims}_{name}")




data_dims = 4096
train_and_save_PCA(data_dims)
data_dims = 1024
train_and_save_PCA(data_dims)