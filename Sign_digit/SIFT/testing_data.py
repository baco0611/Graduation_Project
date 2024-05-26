import joblib
import numpy as np

raw_description = joblib.load("./data/dataset/raw_description.joblib")
negative_description = joblib.load("./data/dataset/negative_description.joblib")
resized_description = joblib.load("./data/dataset/resized_description.joblib")
rotated_description = joblib.load("./data/dataset/rotated_description.joblib")
flipped_description = joblib.load("./data/dataset/flipped_description.joblib")
process_description = joblib.load("./data/dataset/process_description.joblib")

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