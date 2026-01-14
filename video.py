import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture('IMG_7653.mov')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    prediction = model(frame,verbose=False)
    annotated = prediction[0]
    boxes = annotated.boxes.xywh
    areas = boxes[:, 2] * boxes[:, 3]
    largest_idx = areas.argmax()

    annotated.boxes = annotated.boxes[largest_idx]
    annotated.keypoints = annotated.keypoints[largest_idx]


    h, w, c= frame.shape
    resized = cv2.resize(annotated.plot(),(w//2,h//2))
    cv2.imshow('prediction',resized)
    cv2.imshow('original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows
        break