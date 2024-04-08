from ultralytics import YOLO
import cv2
import cvzone
import math
# from sort import *

import keyboard # press 'q' to stop terminal

cap = cv2.VideoCapture('Videos/cars.mp4')

model = YOLO('Yolo_Weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable","toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread('mask.png')

while True:
    success, img = cap.read()
    imgRegin = cv2.bitwise_and(img, mask)
    # img = cv2.flip(img, 1)
    results = model(imgRegin, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2) 
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,200,255), 3)
            # print(x1,y1,x2,y2) 
            
            w,h = x2-x1, y2-y1

            conf = math.ceil(box.conf[0]*100)/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' or currentClass == 'motorbike' and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35,y1)) ,scale=0.6, thickness=1, offset=3)
                cvzone.cornerRect(img, (x1,y1,w,h), l=9 )
            



    if keyboard.is_pressed('q'):
        break

    cv2.imshow('Image', img)
    # cv2.imshow('ImageRegin', imgRegin)
    cv2.waitKey(1)
    