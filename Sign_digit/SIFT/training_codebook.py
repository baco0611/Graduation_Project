from function.bovw import *
import joblib
import time
from config import *

# Start time counting
start_time = time.time()


print("Loading data ...")
description = joblib.load(f'./data/dataset/{name}_description.joblib')
print(len(description))
all_descriptor = description

print("Training codebook ...")
codebook = build_codebook(all_descriptor, size)

print("Save codebook ...")
path_codebook = f'./data/model/{date}_{name}_{size}_codebook.joblib'
save_codebook(codebook, path_codebook)

# End and calculate time
end_time = time.time()
execution_time = end_time - start_time

print("Thời gian thực thi: {:.5f} giây".format(execution_time))