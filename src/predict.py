import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO("runs/detect/train-7/weights/best.pt")
img = "test form my phone\\road deek aljen.jpg"
results = model(img)

#draw results on the original image
annotated_frame = results[0].plot()

#save the annotated image
cv2.imwrite("results/visualizations/output_result.png", annotated_frame)

# show image by OpenCV
cv2.imshow("YOLOv8 Results", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
