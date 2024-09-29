# MINOR-PROJECT
DOG BREED DETECTION 
import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
from extract_bottleneck_features import extract_Resnet50
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
import cv2
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

# Load the dog breed names
with open("breeds.txt", "rb") as file:
    dog_names = pickle.load(file)

# Load the pre-trained ResNet50 model
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=[7, 7, 2048]))
Resnet50_model.add(Dropout(0.3))
Resnet50_model.add(Dense(1024, activation='relu'))
Resnet50_model.add(Dropout(0.4))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

# Function to process an image and detect the breed
def detector(img_path):
    if dog_detector(img_path):
        breed = Resnet50_predict_breed(img_path)
        return f'The predicted dog breed is {breed}'
    else:
        return 'No dog detected'

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def Resnet50_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    name = dog_names[np.argmax(predicted_vector)]
    return name.split('.')[-1]

def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Create and configure the root window
root = tk.Tk()
root.title("Dog Breed Detector")

# Function to open a file dialog and update the selected image and breed name
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Update the selected image in the GUI
        img = Image.open(file_path)
        img.thumbnail((224, 224))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        # Call the detector function and display the result on the GUI
        breed_name = detector(file_path)
        result_label.config(text=breed_name)

# Create and configure GUI elements
upload_button = tk.Button(root, text="Upload Image", command=open_file_dialog)
upload_button.pack(pady=10)

image_label = Label(root)
image_label.pack(padx=10, pady=10)

result_label = Label(root, text="", wraplength=300)
result_label.pack(padx=10, pady=10)

# Run the GUI application
root.mainloop()
