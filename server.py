import io
import os
import uvicorn
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/aksarajawa-s.pt', force_reload=True)
with open("classes/classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Create a dictionary to map class IDs to their corresponding names
class_map = {i: class_names[i] for i in range(len(class_names))}
# Define the endpoint for object detection
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded file into memory as bytes
    contents = await file.read()
    
    # Load the image from bytes
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Run object detection on the image using YOLOv5
    results = model(image)
    
    # Format the results as a JSON response
    objects = []
    for obj in results.xyxy[0]:
        class_id = int(obj[5])
        class_name = class_map[class_id]
        confidence = float(obj[4])
        bbox = obj[:4].tolist()
        objects.append({"class": class_name, "confidence": confidence, "bbox": bbox})
    
    # Format the results as a JSON response
    response = {"objects": objects}
    
    # Return the results as a JSON response
    return JSONResponse(content=response)

def annotate_image(image, results):
    for obj in results.xyxy[0]:
        class_id = int(obj[5])
        class_name = class_map[class_id]
        confidence = float(obj[4])
        bbox = obj[:4].tolist()
        bbox = [int(coord) for coord in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Define a route to handle image uploads and return annotated image
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Load the image from bytes
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Run object detection on the image using YOLOv5
    results = model(image)

    # Annotate the image with bounding boxes
    annotated_image = annotate_image(np.array(image), results)
    
    print("aman")
    # Convert the annotated image to bytes and return
    buffered = io.BytesIO()
    Image.fromarray(annotated_image).save(buffered, format="JPEG")
    return {"image": buffered.getvalue()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))