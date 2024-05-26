import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from VGG8.function.config import *
from VGG8.function.Processing_function import resize_list_image
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import gc
from keras.callbacks import LearningRateScheduler

# Hàm tải dữ liệu
def load_data_from_folder(folder_path):
    return joblib.load(folder_path)

def process_data(images, labels, size):
    data = resize_list_image(images)

    print(type(data))
    print(len(labels))

    data = np.array(data)
    labels = to_categorical(labels, num_classes=size)
    
    return train_test_split(data, labels, test_size=0.3, random_state=42)


# Hàm trộn dữ liệu từ nhiều folder
def mix_data(folders, label):
    images = []
    for folder in folders:
        images += load_data_from_folder(folder)
    # return images, [label] * len(images)
    return images, [label] * len(images)


# Hàm xây dựng mô hình VGG11
def build_vgg11_model(input_shape=(224, 224, 3), num_classes=2):

    #VGG8
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.summary()

    return model

def plot_history(history, model_name):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy', color='red')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./image/{model_name}_loss_accuracy_plot.png")
    plt.close()

# Hàm đánh giá mô hình và vẽ confusion matrix
def evaluate_and_confusion_matrix(model, test_x, test_y, model_name):

    dataset = tf.data.Dataset.from_tensor_slices(test_x).batch(1000)

    gc.collect()

    # Dự đoán theo batch sử dụng dataset
    predictions = []
    for batch_data in dataset:
        batch_predictions = model.predict(batch_data)
        predictions.extend(batch_predictions)

    test_predictions = predictions

    # predictions = np.array(predictions)
    # test_predictions = model.predict(test_x)
    test_predictions_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = np.argmax(test_y, axis=1)
    
    conf_matrix = confusion_matrix(test_true_labels, test_predictions_labels)
    accuracy = accuracy_score(test_true_labels, test_predictions_labels)
    print("Test accuracy: ", accuracy)

    
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
    plt.xlabel('Predicted labels', fontsize="14")
    plt.ylabel('True labels', fontsize="14")
    plt.title('Confusion Matrix', fontsize="14")
    plt.savefig(f"./image/{model_name}_confusion_matrix.png")
    plt.close()
    
    del dataset, batch_data, predictions, test_predictions, test_predictions_labels, test_true_labels  # Giải phóng bộ nhớ sau khi dự đoán và đánh giá
    gc.collect()
    tf.keras.backend.clear_session()

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def train_and_save_model(train_x, train_y, test_x, test_y, model_name, epochs=30, unit = 2):
    model = build_vgg11_model(num_classes=unit)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=["accuracy"])

    print(f"Training {model_name} model ...")

    callback = LearningRateScheduler(scheduler)

    history = model.fit(train_x, train_y, batch_size=8, epochs=epochs)
    
    plot_history(history, model_name)
    model.save(f"./model/{model_name}_CNN_model.h5")
    del train_x, train_y, history  # Giải phóng bộ nhớ sau khi huấn luyện và đánh giá
    gc.collect()
    tf.keras.backend.clear_session()
    
    evaluate_and_confusion_matrix(model, test_x, test_y, model_name)


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
size = len(set(labels))

train_x, test_x, train_y, test_y = process_data(data, labels, size)
# train_x, test_x = train_x / 255.0, test_x / 255.0
print(len(train_x), len(test_x))
print(len(train_y), len(test_y))
print(type(train_x), type(test_x))

del data, labels
gc.collect()

model_name = f"{date}_VGG8_{name}_{num_of_epoch}"
train_and_save_model(train_x, train_y, test_x, test_y, model_name, epochs=num_of_epoch, unit = size)

del train_x
gc.collect()
del test_x
gc.collect()
