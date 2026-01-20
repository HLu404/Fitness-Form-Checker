from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse
from .models import Video
from .forms import VideoUploadForm
import cv2
from ultralytics import YOLO
from .utils import video_processing as vp
import torch

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

pose_model = YOLO("yolov8n-pose.pt")
outline_model = YOLO("yolov8n-seg.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pose_model = YOLO("yolov8n-pose.pt").to(DEVICE)
outline_model = YOLO("yolov8n-seg.pt").to(DEVICE)


# ---------------- EXERCISE MAP ----------------
EXERCISES = [
    "deadlift",
    "squat",
    "lat",
    "pul",
    "dip"
]

FORM_CHECKS = {
    "deadlift": vp.deadlift_form,
    "squat": vp.squat_form,
    "lat": vp.lat_form,
    "pul": vp.pul_form,
    "dip": vp.dips_form
}

# ---------------- STREAM GENERATOR ----------------
def exercise_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    detected_exercise = False
    exercise_checked = ""
    counts = {ex: 0 for ex in EXERCISES}

    starting_rep = True
    last_rep_angle = 0
    current_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_pose, idx, person_detected = vp.predict(pose_model, frame)

        feedback = []

        if person_detected:
            pose_coords = annotated_pose.keypoints[idx].data.cpu().numpy()

            # ---------- EXERCISE DETECTION ----------
            if not detected_exercise:
                detected = vp.exercise_detection(pose_coords)

                for ex in EXERCISES:
                    if detected == ex:
                        counts[ex] += 1
                    else:
                        counts[ex] == 0

                    if counts[ex] >= fps // 2:
                        exercise_checked = ex
                        detected_exercise = True
                        break

                feedback.append(("Detecting exercise...", (255, 255, 0)))

            # ---------- FORM CHECK ----------
            else:
                feedback.append((f"Exercise: {exercise_checked}", (0, 255, 0)))

                if exercise_checked in ["deadlift", "squat"]:
                    annotated_outline, oidx, _ = vp.predict(outline_model, frame)

                    # ONLY call deadlift/squat if outline exists
                    if annotated_outline.masks is not None:
                        outline_coords = annotated_outline.masks.xy[oidx]
                        feedback.extend(
                            FORM_CHECKS[exercise_checked](
                                frame, pose_coords, outline_coords
                            )
                        )
                    else:
                        feedback.append(("Waiting for body outline...", (255, 255, 0)))

                else:
                    feedback.extend(
                        FORM_CHECKS[exercise_checked](
                            frame, pose_coords
                        )
                    )

        # ---------- DRAW FEEDBACK ----------
        y = 40
        for text, color in feedback:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y += 35

        # ---------- STREAM ----------
        _, jpeg = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

    cap.release()

# ---------- VIDEO FEED ----------
def video_feed(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    return StreamingHttpResponse(
        exercise_stream(video.original.path),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )

# ---------- STREAM PAGE ----------
def stream(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    return render(request, "stream.html", {"video": video})