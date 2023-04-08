import argparse
import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/aksarajawa-m.pt')
model.conf = 0.5

# Load the class names
with open("classes/classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Function to detect objects and draw bounding boxes on the image
def detect_objects(input_path):
    # Load the image
    img = cv2.imread(input_path)

    # Run the YOLOv5 model to detect objects
    results = model(img)
    results.show()

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="path to the input image")
args = parser.parse_args()

# Call the detect_objects function with the input and output paths from the command line arguments
detect_objects(args.input_path)
