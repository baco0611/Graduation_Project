from function.image_processing import *
import time
import joblib

# Start time counting
start_time = time.time()

def extract_function(name):
    #Load files
    print(f"\nLoading file from {name} ...")
    image_dir = f"../dataset/data/{name}_image.joblib"
    (_, gray) = load_images(image_dir)

    print(len(gray))

    #Extract SIFT features
    print("Extracting feature ...")
    (keypoints, description) = extract_visual_features(gray)

    print(len(description))

    print("Saving data ...")
    joblib.dump(description, f"./data/dataset/{name}_description.joblib")


names = ["process", "raw", "negative", "flipped", "rotated", "resized"]
for name in names:
    extract_function(name)
# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))