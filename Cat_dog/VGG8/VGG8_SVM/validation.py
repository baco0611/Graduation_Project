import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from VGG8.function.config import *
from VGG8_SVM_config import *
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def validate(name, index):
    data = joblib.load(folders[index])
    print(len(data), len(labels))

    predictions = model.predict(data)
    conf_matrix = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    print(f"{name} accuracy:", accuracy)
        
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.savefig(f"./image/validation/{model_name}_{name}_confuse_matrix.png")
    plt.close()

directory = "./image/validation"
if not os.path.exists(directory):
    os.makedirs(directory)

data_folder = f"{feature_dims}dims_data"
model_name = f"{date}_SVM_{feature_dims}"
print("Validate model", model_name)

folders = [
    f"./data/dataset/{data_folder}/raw_image_features.joblib",
    f"./data/dataset/{data_folder}/negative_image_features.joblib",
    f"./data/dataset/{data_folder}/resized_image_features.joblib",
    f"./data/dataset/{data_folder}/rotated_image_features.joblib",
    f"./data/dataset/{data_folder}/flipped_image_features.joblib",
]

model = joblib.load("./data/model/" + model_name + ".joblib")
labels = joblib.load("../.././dataset/data/label.joblib")

i = 0
for x in ["raw", "negative", "resized", "rotated", "flipped"]:
    validate(x, i)
    i+=1