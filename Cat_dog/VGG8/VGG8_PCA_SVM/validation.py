from PCA_SVM_config import *
import os
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def validate(feature_dims, cnn_dims, name, model, index):
    print(cnn_dims, feature_dims, name)

    folders = [
        f"./data/dataset/{feature_dims}/{cnn_dims}_raw_image_features.joblib",
        f"./data/dataset/{feature_dims}/{cnn_dims}_negative_image_features.joblib",
        f"./data/dataset/{feature_dims}/{cnn_dims}_resized_image_features.joblib",
        f"./data/dataset/{feature_dims}/{cnn_dims}_rotated_image_features.joblib",
        f"./data/dataset/{feature_dims}/{cnn_dims}_flipped_image_features.joblib",
    ]

    labels = joblib.load("../.././dataset/data/label.joblib")
    data = joblib.load(folders[index])
    print(len(data), len(labels))
    if index == 5:
        print(index)
        new_labels *= 5

    predictions = model.predict(data)
    conf_matrix = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    print(f"{name} accuracy:", accuracy)
        
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.savefig(f"./image/validation/{model_name}_{name}_confuse_matrix.png")
    plt.close()


directory = "./image/validation"
if not os.path.exists(directory):
    os.makedirs(directory)

for x in range(200, 600, 100):
    model_name = f"{date}_SVM_{cnn_dims}_{x}"
    model = joblib.load("./data/model/" + model_name + ".joblib")
    i = 0

    print("\n\n\n")
    print(cnn_dims, x)

    for y in ["raw", "negative", "resized", "rotated", "flipped"]:
        validate(x, cnn_dims, y, model, i)
        i+=1