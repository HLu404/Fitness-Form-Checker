import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def yolo_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        _, jpeg = cv2.imencode('.jpg', annotated)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            jpeg.tobytes() +
            b'\r\n'
        )

    cap.release()
