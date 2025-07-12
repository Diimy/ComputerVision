import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([DetectionModel])  # autoriza carregar o modelo

from ultralytics import YOLO
import cv2

model = YOLO("yolov8l.pt")  # Load a pretrained YOLOv8 model
results = model("images/car01.jpg", show=True)
cv2.waitKey(0)
