import cv2
import numpy as np
from math import sqrt
from math import acos
from ultralytics import YOLO
import video

pose = YOLO('yolov8n-pose.pt')
pose.to('cuda')
outline = YOLO('yolov8n-seg.pt')
outline.to('cuda')
cap = cv2.VideoCapture('Deadlift.mov')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("Output.mp4", fourcc, fps, (width, height))
exercise = ['Deadlift','Squat','Lat Pulldowns','Pullups','Chest Dips']
exercise_checked = ""
counts = {ex: 0 for ex in exercise}
#checks = {'deadlift': video.deadlift_form,'squat':video.squat_form, 
#          'lat':video.lat_form,'pul':video.pull_form,'dip':video.dip_form}
checks = {'Lat Pulldowns':video.lat_form,'Pullups':video.pul_form,'Chest Dips':video.dips_form,
          'Deadlift':video.deadlift_form,'Squat':video.squat_form}
detected_exercise = False
starting_rep = True
last_rep_angle = 0
results = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_pose, largest_idx_pose, person_detected = video.predict(pose,frame)

    if person_detected:

        pose_coords = annotated_pose.keypoints[largest_idx_pose].data.cpu().numpy() # get the pose coordinates of biggest box 
        
        if not detected_exercise: 
            detected = video.exercise_detection(pose_coords)
            for ex in exercise:
                if detected == ex:
                    counts[ex]+=1
                if counts[ex] == fps//2:
                    exercise_checked = ex
                    detected_exercise = True 
                    break
                else: 
                    counts[ex] = 0

        else:
            cv2.putText(frame,f'Exercise: {ex}',(0,75),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if ex == "Deadlift" or ex == "Squat":
                annotated_outline, largest_idx_outline, person_detected = video.predict(outline,frame)
                if annotated_outline.masks is not None:
                    outline_coords = annotated_outline.masks.xy[largest_idx_outline]
                    if last_rep_angle - results>=5:
                        starting_rep = True
                    else: 
                        starting_rep = False
                    last_rep_angle = results
                    results = checks[ex](frame,out,pose_coords,outline_coords,starting_rep) 
                else:
                    out.write(frame)  # skip frame if no mask
            else: 
                if last_rep_angle - results >=5:
                    starting_rep = True
                else: 
                    starting_rep = False
                last_rep_angle = results
                results = checks[ex](frame,out,pose_coords, starting_rep)

print("done")
cap.release()
out.release()