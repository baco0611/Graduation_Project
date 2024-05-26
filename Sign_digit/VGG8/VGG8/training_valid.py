import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from VGG8.function.Processing_function import resize_list_image
from VGG8.function.config import *
import os
import joblib
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import gc
import shutil

def process_data(images):
    data = resize_list_image(images)
    data = np.array(data)
    
    return data


folders = [
    "../.././dataset/data/raw_image.joblib",
    "../.././dataset/data/negative_image.joblib",
    "../.././dataset/data/resized_image.joblib",
    "../.././dataset/data/rotated_image.joblib",
    "../.././dataset/data/flipped_image.joblib",
    "../.././dataset/data/process_image.joblib",
]

data = joblib.load(folders[data_num])
labels = joblib.load("../.././dataset/data/label.joblib")
if data_num == 5:
    labels = labels * 4
labels = to_categorical(labels, num_classes=len(set(labels)))
print(len(data), len(labels))
batch_size = 1000

# Tạo thư mục tạm để lưu các batch
temp_dir = "./temp_batches"
os.makedirs(temp_dir, exist_ok=True)

# Chia dữ liệu thành các batch và lưu vào thư mục tạm
num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
for i in range(num_batches):
    batch_data = data[i*batch_size:(i+1)*batch_size]
    batch_labels = labels[i*batch_size:(i+1)*batch_size]
    batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
    joblib.dump((batch_data, batch_labels), batch_file)

# Giải phóng bộ nhớ
del data
del labels
gc.collect()

# Load model
model_name = f"{date}_VGG8_{name}_{num_of_epoch}"
model = load_model(f"./model/{model_name}_CNN_model.h5")

# Load từng batch từ thư mục tạm để tính toán và dự đoán
predictions = []
true_labels = []

print("\n\n")
for i in range(num_batches):
    batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
    batch_data, batch_labels = joblib.load(batch_file)

    batch_data = process_data(batch_data)

    batch_predictions = model.predict(batch_data)
    predictions.extend(batch_predictions)
    true_labels.extend(batch_labels)
    # Giải phóng bộ nhớ sau mỗi batch
    del batch_data
    del batch_labels
    gc.collect()

predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Chuyển đổi dự đoán và nhãn thực tế từ dạng one-hot về dạng chỉ số
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(true_labels, axis=1)

# print(predicted_labels[:10])
# print(true_labels[:10])

# Tính toán confusion matrix và accuracy
conf_matrix = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
print("Model accuracy: ", accuracy)

# Vẽ và hiển thị confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
plt.xlabel('Predicted labels', fontsize="14")
plt.ylabel('True labels', fontsize="14")
plt.title('Confusion Matrix', fontsize="14")
plt.savefig(f"./image/{model_name}.png")
plt.show()

# Xóa thư mục tạm
shutil.rmtree(temp_dir)

# Giải phóng bộ nhớ
tf.keras.backend.clear_session()
gc.collect()