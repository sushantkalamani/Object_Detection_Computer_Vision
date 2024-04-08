from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('Videos/cars.mp4')

model = YOLO('Yolo_Weights/yolov8l.pt')

classes_to_count = [2]
unique_car_ids = set()

mask = cv2.imread('mask.png')

while True:
    success, img = cap.read()
    imgRegin = cv2.bitwise_and(img, mask)
    # img = cv2.flip(img, 1)
    results = model.track(imgRegin, persist=True, classes=classes_to_count )

    boxes = results[0].boxes.xywh.cpu()
    try:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    except AttributeError:
        print("No detections found or 'id' attribute missing in boxes data structure.")

    # if track_ids:
    # Count cars based on unique track IDs
    for box, track_id in zip(boxes, track_ids):
        if track_id not in unique_car_ids:
            unique_car_ids.add(track_id)

    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Car Count: {len(unique_car_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Image', annotated_frame)
    # cv2.imshow('ImageRegin', imgRegin)
    cv2.waitKey(1)