# imports
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml

if __name__ == '__main__':

    # load config
    with open("configs/config.yaml","r") as file:
        config = yaml.safe_load(file)
    print("✅ config loaded")

    epochs = config.get("training", {}).get("epochs")
    batch = config.get("training", {}).get("batch_size")
    imgsz = config.get("data", {}).get("img_size")
    device = config.get("training", {}).get("device")
    patience = config.get("training", {}).get("patience")

    # load model
    model = YOLO("runs/detect/train-7/weights/best.pt")
    print("✅ model loaded")

    #train model
    print("🚀 starting training...")
    results = model.train(
    data="data/raw/data.yaml",
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    device=device,
    workers=0,
    cache=False,
    resume=False,
    patience=patience   
)

    print("✅ done!")

