import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import cv2
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="Khalil200383/RoadGuard-AI",
    filename="best.pt"
)
model = YOLO(model_path)

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
 