from VGG8_SVM_config import *
from sklearn import svm
import joblib
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(data, labels):

    data = np.array(data)
    
    return train_test_split(data, labels, test_size=0.3, random_state=42)


def train_and_save_model(train_x, train_y, test_x, test_y, model_name):
    print("Training model ...")

    model = svm.SVC(kernel='linear')
    model.fit(train_x, train_y)

    joblib.dump(model, "./data/model/" + model_name + ".joblib")

    evaluate_and_confusion_matrix(model, test_x, test_y, model_name)

    return model

# Hàm đánh giá mô hình và vẽ confusion matrix
def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):
    test_predictions = model.predict(test_x)

    
    conf_matrix = confusion_matrix(test_y, test_predictions)
    accuracy = accuracy_score(test_y, test_predictions)
    print("Test Accuracy:", accuracy)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.savefig(f"./image/{model_name}_confusion_matrix.png")
    plt.close()



data_folder = f"{feature_dims}dims_data"
model_name = f"{date}_SVM_{feature_dims}"
print("Training model", model_name)


folders = [
    f"./data/dataset/{data_folder}/raw_image_features.joblib",
    f"./data/dataset/{data_folder}/negative_image_features.joblib",
    f"./data/dataset/{data_folder}/resized_image_features.joblib",
    f"./data/dataset/{data_folder}/rotated_image_features.joblib",
    f"./data/dataset/{data_folder}/flipped_image_features.joblib",
    f"./data/dataset/{data_folder}/process_image_features.joblib",
]


data = joblib.load(folders[data_num])
labels = joblib.load("../.././dataset/data/label.joblib")
if data_num == 5:
    labels = labels * 4

print(len(data), len(labels))

train_x, test_x, train_y, test_y = process_data(data, labels)
print(len(train_x), len(train_y))
print(len(test_x), len(test_y))

model = train_and_save_model(train_x, train_y, test_x, test_y, model_name)

all_data = []
all_data.extend(train_x)
all_data.extend(test_x)
all_data = np.array(all_data)

all_labels = []
all_labels.extend(train_y)
all_labels.extend(test_y)
all_labels = np.array(all_labels)

print(type(all_data), len(all_data))
print(type(all_labels), len(all_labels))

predictions = model.predict(all_data)
conf_matrix = confusion_matrix(all_labels, predictions)
accuracy = accuracy_score(all_labels, predictions)
print("Model Accuracy:", accuracy)
    
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f"./image/{model_name}.png")
plt.close()