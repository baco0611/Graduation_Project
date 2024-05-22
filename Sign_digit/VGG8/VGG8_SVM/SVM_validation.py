from sklearn import svm
import joblib
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from Processing_function import resize_list_image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(cases):
    data = []
    labels = []
    for (images, label) in cases:
        data.extend(images)
        labels.extend(label)

    print(type(data))
    print(data[0])
    print(len(labels))

    data = np.array(data)
    # labels = to_categorical(labels, num_classes=2)
    
    # return train_test_split(data, labels, test_size=0.3, random_state=42)
    return data, labels


# Hàm trộn dữ liệu từ nhiều folder
def mix_data(folders, label):
    images = []
    for folder in folders:
        data = load_data_from_folder(folder)
        if len(data) > 0:  # Kiểm tra xem thư mục có chứa ít nhất một hình ảnh không
            images.append(data)  # Chỉ thêm dữ liệu vào danh sách nếu thư mục không trống

    images = np.array(images)
    images = images.squeeze()
    return images, [label] * len(images)

feature_dims = 1024
data_folder = f"extracted_{feature_dims}dims_data"
date = "20240521"
model_name = f"{date}_SVM_{feature_dims}"

folders = [
    f"./data/{data_folder}/raw_image_features.joblib",
    f"./data/{data_folder}/negative_image_features.joblib",
    f"./data/{data_folder}/resized_image_features.joblib",
    f"./data/{data_folder}/rotated_image_features.joblib",
    f"./data/{data_folder}/flipped_image_features.joblib",
]

#Lấy dữ liệu mong muốn
# Hàm mix_data với hai tham số ([array of dataset], label)
model = joblib.load("./data/" + model_name + ".joblib")

labels = joblib.load("../dataset/data/label.joblib")
labels = [x for x in labels]
size = len(set(labels))

# Hàm mix_data với hai tham số ([array of dataset], label)

def validate(name, index):
    data = joblib.load(folders[index])
    print(len(data), len(labels))

    predictions = model.predict(data)
    conf_matrix = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    print(f"{name} Accuracy:", accuracy)
        
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.xticks(fontsize="14")
    plt.yticks(fontsize="14")
    plt.savefig(f"./img/validation/{model_name}_{name}_confuse_matrix.png")
    plt.close()

i = 0
for x in ["raw", "negative", "resized", "rotated", "flipped"]:
    validate(x, i)
    i+=1