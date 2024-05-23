import joblib
import numpy as np

def test_data(cnn_dims, feature_dims):
    print(f"\n\n{cnn_dims} to {feature_dims} data testing")
    raw_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_raw_image_features.joblib")
    negative_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_negative_image_features.joblib")
    resized_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_resized_image_features.joblib")
    rotated_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_rotated_image_features.joblib")
    flipped_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_flipped_image_features.joblib")
    process_description = joblib.load(f"./data/dataset/{feature_dims}/{cnn_dims}_process_image_features.joblib")

    print(type(raw_description))
    print(type(raw_description[0]))

    diff = 0
    for i in range(15000):
        if not (np.array_equal(raw_description[i], process_description[i])):
            diff += 1
    print("Error between raw and process:", diff)

    diff = 0
    for i in range(15000):
        if not (np.array_equal(negative_description[i], process_description[i + 15000])):
            diff += 1
    print("Error between neg and process:", diff)

    diff = 0
    for i in range(15000):
        if not (np.array_equal(resized_description[i], process_description[i + 15000 * 2])):
            diff += 1
    print("Error between rez and process:", diff)

    diff = 0
    for i in range(15000):
        if not (np.array_equal(rotated_description[i], process_description[i + 15000 * 3])):
            diff += 1
    print("Error between rot and process:", diff)

    diff = 0
    for i in range(15000):
        if not (np.array_equal(flipped_description[i], process_description[i + 15000 * 4])):
            diff += 1
    print("Error between flip and process:", diff)

for x in [1024, 4096]:
    for y in range(200, 600, 100):
        test_data(x, y)