import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("runs/detect/train-7/weights/best.pt")

# predict function
def predict(image):
    results = model(image)
    annotated_image = results[0].plot()
    return annotated_image
# demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="numpy"), 
                    outputs=gr.Image(type="numpy"),
                    title="Identifying road defects Demo",
                    description="Upload an image to see the detected objects.")
demo.launch()
