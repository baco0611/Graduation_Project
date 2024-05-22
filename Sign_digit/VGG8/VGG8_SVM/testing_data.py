import joblib
import numpy as np

print("1024dims data testing")
raw_description = joblib.load("./data/dataset/1024dims_data/raw_image_features.joblib")
negative_description = joblib.load("./data/dataset/1024dims_data/negative_image_features.joblib")
resized_description = joblib.load("./data/dataset/1024dims_data/resized_image_features.joblib")
rotated_description = joblib.load("./data/dataset/1024dims_data/rotated_image_features.joblib")
flipped_description = joblib.load("./data/dataset/1024dims_data/flipped_image_features.joblib")
process_description = joblib.load("./data/dataset/1024dims_data/process_image_features.joblib")

print("Length of raw", len(raw_description))
print("Length of negative", len(negative_description))
print("Length of resized", len(resized_description))
print("Length of rotated", len(rotated_description))
print("Length of flipped", len(flipped_description))
print("Length of process", len(process_description))

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


print("\n\n\n4096dims data testing")
raw_description = joblib.load("./data/dataset/4096dims_data/raw_image_features.joblib")
negative_description = joblib.load("./data/dataset/4096dims_data/negative_image_features.joblib")
resized_description = joblib.load("./data/dataset/4096dims_data/resized_image_features.joblib")
rotated_description = joblib.load("./data/dataset/4096dims_data/rotated_image_features.joblib")
flipped_description = joblib.load("./data/dataset/4096dims_data/flipped_image_features.joblib")
process_description = joblib.load("./data/dataset/4096dims_data/process_image_features.joblib")

print("Length of raw", len(raw_description))
print("Length of negative", len(negative_description))
print("Length of resized", len(resized_description))
print("Length of rotated", len(rotated_description))
print("Length of flipped", len(flipped_description))
print("Length of process", len(process_description))

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