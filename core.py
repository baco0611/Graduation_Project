import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog
import cv2
import numpy as np

def calculate_accuracy(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not read image")
        return None

    # Assume the confusion matrix is located in a fixed area, adjust values as needed
    x, y, w, h = 50, 50, 900, 800  # Change these values to fit your image
    confusion_matrix = img[y:y+h, x:x+w]

    # Thresholding to isolate the matrix values
    ret, thresh = cv2.threshold(confusion_matrix, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_sum = 0
    diagonal_sum = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        value = int(img[y:y+h, x:x+w].mean())
        if x == y:  # Check if the value is on the diagonal
            diagonal_sum += value
        total_sum += value
    
    accuracy = diagonal_sum / total_sum if total_sum != 0 else 0
    return accuracy

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        show_accuracy(file_path)

def drop(event):
    file_path = event.data.strip('{}')
    if file_path:
        show_accuracy(file_path)

def show_accuracy(file_path):
    accuracy = calculate_accuracy(file_path)
    if accuracy is not None:
        result_label.config(text=f"Accuracy: {accuracy:.2%}")

# Create the GUI
root = TkinterDnD.Tk()
root.title("Confusion Matrix Accuracy Calculator")
root.geometry("400x300")
root.minsize(200, 200)

frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

open_button = tk.Button(frame, text="Open Image", command=open_image)
open_button.pack(pady=10)

drop_label = tk.Label(frame, text="Or drag and drop an image file here")
drop_label.pack(pady=10)

result_label = tk.Label(frame, text="Accuracy: ")
result_label.pack(pady=10)

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

root.mainloop()
