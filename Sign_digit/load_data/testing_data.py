import joblib
import numpy as np

raw_image = joblib.load("../dataset/data/raw_image.joblib")
negative_image = joblib.load("../dataset/data/negative_image.joblib")
resized_image = joblib.load("../dataset/data/resized_image.joblib")
rotated_image = joblib.load("../dataset/data/rotated_image.joblib")
flipped_image = joblib.load("../dataset/data/flipped_image.joblib")
process_image = joblib.load("../dataset/data/process_image.joblib")

print("Length of raw", len(raw_image))
print("Length of negative", len(negative_image))
print("Length of resized", len(resized_image))
print("Length of rotated", len(rotated_image))
print("Length of flipped", len(flipped_image))
print("Length of process", len(process_image))

diff = 0
for i in range(15000):
    if not (np.array_equal(raw_image[i], process_image[i])):
        diff += 1
print("Error between raw and process:", diff)

diff = 0
for i in range(15000):
    if not (np.array_equal(negative_image[i], process_image[i + 15000])):
        diff += 1
print("Error between neg and process:", diff)

diff = 0
for i in range(15000):
    if not (np.array_equal(resized_image[i], process_image[i + 15000 * 2])):
        diff += 1
print("Error between rez and process:", diff)

diff = 0
for i in range(15000):
    if not (np.array_equal(rotated_image[i], process_image[i + 15000 * 3])):
        diff += 1
print("Error between rot and process:", diff)