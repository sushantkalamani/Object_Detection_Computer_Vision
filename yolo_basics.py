from ultralytics import YOLO
import cv2

model = YOLO('Yolo_Weights\yolov8n.pt')

results = model('Images/3.jpeg', show = True)
cv2.waitKey(0)
