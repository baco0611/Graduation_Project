from function.bovw import *
from config import *
import joblib
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Start time counting
start_time = time.time()

print("Loading data ...")
description = joblib.load(f'./data/dataset/{name}_description.joblib')
codebook = joblib.load(f"./data/model/{date}_{name}_{size}_codebook.joblib")
labels = joblib.load("../dataset/data/label.joblib")
process_labels = labels * 4
labels = process_labels

print("Số lượng mẫu", len(description))
print("Số lượng nhãn", len(labels))

print("Represent data ...")
data = [represent_image_features(x, codebook) for x in description]

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.3, random_state=42)

print("Training model ...")

model = svm.SVC(kernel='linear')  # Chọn kernel tùy ý (linear, rbf, ...)
model.fit(train_x, train_y)

print("\nSaving model ...") 
model_name = f"{date_svm}_{name}_{size}"
joblib.dump(model, f"./data/model/SVM_{model_name}_model.joblib")

print("\nAccuracy of Valid set and Test set")


# Dự đoán trên tập test
test_preds = model.predict(test_x)
test_accuracy = accuracy_score(test_y, test_preds)
print("Test Accuracy:", test_accuracy)
# Tính toán confusion matrix
conf_matrix = confusion_matrix(test_y, test_preds)

plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f'./image/{model_name}_confusion_test_matrix.png')
plt.show()


# Dự đoán trên toàn bộ tập dữ liệu
data_preds = model.predict(data)
accuracy = accuracy_score(labels, data_preds)
print("Dataset Accuracy:", accuracy)
conf_matrix = confusion_matrix(labels, data_preds)

plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f'./image/{model_name}_confusion_dataset_matrix.png')
plt.show()

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))