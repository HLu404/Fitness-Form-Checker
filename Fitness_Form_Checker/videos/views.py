from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse
from .models import Video
from .forms import VideoUploadForm

import cv2
from ultralytics import YOLO


# ---------- HOME ----------
def homepage(request):
    return render(request, "home.html")


# ---------- UPLOAD ----------
def upload(request):
    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            return redirect("videos:stream", video_id=video.id)
    else:
        form = VideoUploadForm()

    return render(request, "upload.html", {"form": form})


# ---------- YOLO MODEL ----------
model = YOLO("yolov8n-pose.pt")


# ---------- FRAME GENERATOR ----------
def yolo_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        _, jpeg = cv2.imencode(".jpg", annotated)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

    cap.release()


# ---------- VIDEO FEED (RAW STREAM) ----------
def video_feed(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    return StreamingHttpResponse(
        # yolo_stream(video.file.path),
        yolo_stream(video.original.path),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------- STREAM PAGE ----------
def stream(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    return render(request, "stream.html", {"video": video})
