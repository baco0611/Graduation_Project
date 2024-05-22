import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from VGG8.function.Processing_function import resize_list_image
from VGG8.function.config import *
import joblib
from keras.models import Model, load_model
import os
import tensorflow as tf
import gc
import shutil
import numpy as np

def define_extract_model(model_name, dims):
    model_path = "../VGG8/model/" + model_name + "_CNN_model.h5"
    model = load_model(model_path)

    if dims == 4096:
        index = 11
    else:
        index = 13

    output = [ model.layers[index].output ]
    model = Model(inputs=model.inputs, outputs=output)

    return model

def processing_data(folders):
    images = joblib.load(folders)
    return images

def extract_feature(folders, dims):
    data = processing_data(folders)
    temp_dir = "./temp_batches"
    os.makedirs(temp_dir, exist_ok=True)
    batch_size = 5000

    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch_data = data[i*batch_size:(i+1)*batch_size]
        batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
        joblib.dump(batch_data, batch_file)
    
    del data
    gc.collect()

    model_name = f"{date}_VGG8_{name}_{num_of_epoch}"
    model = define_extract_model(model_name, dims)

    predictions = []
    for i in range(num_batches):
        batch_file = os.path.join(temp_dir, f"batch_{i}.joblib")
        batch_data = joblib.load(batch_file)

        batch_data =  resize_list_image(batch_data)

        batch_predictions = model.predict(batch_data)
        predictions.extend(batch_predictions)
        # Giải phóng bộ nhớ sau mỗi batch
        del batch_data
        gc.collect()

    shutil.rmtree(temp_dir)
    features_vectors = np.array(predictions)
    print(len(features_vectors))
    print(len(features_vectors[0]))

    return features_vectors


folders = [
    "../.././dataset/data/raw_image.joblib",
    "../.././dataset/data/negative_image.joblib",
    "../.././dataset/data/resized_image.joblib",
    "../.././dataset/data/rotated_image.joblib",
    "../.././dataset/data/flipped_image.joblib",
    "../.././dataset/data/process_image.joblib",
]


def extract(output_dir, dims):
    for folder in folders:
        print("\n\nWorking on", folder)
        features = extract_feature(folder, dims)
        filename, _ = os.path.splitext(os.path.basename(folder))
        output_path = os.path.join(output_dir, filename + '_features.joblib')
        joblib.dump(features, output_path)
        print(f"Saved features for {folder} to {output_path}")

dims = 1024
output_dir = f'./data/dataset/{dims}dims_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
extract(output_dir, dims)


dims = 4096
output_dir = f'./data/dataset/{dims}dims_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
extract(output_dir, dims)