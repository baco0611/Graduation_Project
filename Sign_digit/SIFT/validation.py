from function.image_processing import *
from function.bovw import *
from config import *
import time
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading file ...")

model_name = f"{date_svm}_{name}_{size}"

codebook = joblib.load(f"./data/model/{date}_{name}_{size}_codebook.joblib")
model = joblib.load(f"./data/model/SVM_{model_name}_model.joblib")
labels = joblib.load("../dataset/data/label.joblib")

def validation(valid_name):
    print(f"Working on {valid_name} data")
    data = joblib.load(f"./data/dataset/{valid_name}_description.joblib")

    data = [represent_image_features(x, codebook) for x in data]

    y_pred = model.predict(data)

    conf_matrix = confusion_matrix(labels, y_pred)
    test_accuracy = accuracy_score(labels, y_pred)
    print(f"{valid_name} accuracy:", test_accuracy)
    print()

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(f"./image/validation/{model_name}_" + valid_name + "_confuse_matrix.png")


directory = "./image/validation"
if not os.path.exists(directory):
    os.makedirs(directory)

print("Validation ...")
for x in ["raw", "negative", "resized", "rotated", "flipped"]:
    validation(x)