import cv2
import numpy as np
from math import sqrt
from math import acos
from ultralytics import YOLO

# predict using a given model and image. 
def predict(model, image):
    prediction = model(image,verbose=False,show_boxes=False,classes=[0],retina_masks=True)
    annotated = prediction[0]
    if len(annotated.boxes)==0:
        return annotated,None, False
    boxes = annotated.boxes.xywh
    areas = boxes[:,2]*boxes[:,3]
    largest_idx = areas.argmax()
    return annotated, largest_idx, True

# given left joints and right joints, find which group has highest average confidence
def highest_confid(p1,p2,p3,p4,p5,p6):
    avg_conf1 = (p1[2]+p2[2]+p3[2])/3
    avg_conf2 = (p4[2]+p5[2]+p6[2])/3
    if avg_conf1 > avg_conf2 :
        return True
    else:
        return False

# given 3 points a, b, c produces angle abc
def angle(p1,p2,p3):
    C = sqrt( (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
    A = sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    B = sqrt( (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
    angle = acos( (C^2 - A^2 - B^2) / (-2*A*B) )
    return angle 

# returns the slope and y-intercept given by 2 points in y = mx + b format 
def line(p1, p2):
    slope = ( (p2[1]-p1[1]) / (p2[0]-p1[0]) )
    b = p1[1] - slope * p1[0] 
    return slope, b

# returns the perpendicular to the line given by 2 points p1, p2 and passing through p3
def perpendicular(p1, p2, p3):
    slope_i, b_i = line(p1, p2)
    slope_perp = -1/slope_i
    b_perp = p3[1] - slope_perp*p3[0]
    return slope_perp, b_perp

# returns the midpoint given by 2 points p1, p2
def midpoint(p1, p2):
    x_1 = (p1[0]+p2[0])/2
    y_1 = (p1[1]+p2[1])/2
    return x_1, y_1 

# returns if the back of a person is straight in the form they are doing currently 
def straight_back(pose_p, outline_p):
    if highest_confid(pose_p[0][4],pose_p[0][6],pose_p[0][12],
                      pose_p[0][3],pose_p[0][5],pose_p[0][11]):
        p1 = pose_p[0][6]
        p2 = pose_p[0][12]
        left = False # right shoulder, right hip 
    else:
        p1 = pose_p[0][5]
        p2 = pose_p[0][11]
        left = True # left shoulder, left hip
    slope_perp_p1, b_perp_p1 = perpendicular(p1,p2,p1)
    slope_perp_p2, b_perp_p2 = perpendicular(p1,p2,p2)
    if left:
        backp1 = outline_p[np.isclose(outline_p[:,1], slope_perp_p1*outline_p[:,0] + b_perp_p1,atol=1) & 
                        (outline_p[:, 0] < p1[0]) & 
                        (outline_p[:, 1] > p1[1])]
        backp2 = outline_p[np.isclose(outline_p[:,1], slope_perp_p2*outline_p[:,0] + b_perp_p2,atol=1) & 
                        (outline_p[:, 0] < p1[0]) & 
                        (outline_p[:, 1] > p1[1])]
    else:
        backp1 = outline_p[np.isclose(outline_p[:,1], slope_perp_p1*outline_p[:,0] + b_perp_p1,atol=1) & 
                        (outline_p[:, 0] > p1[0]) & 
                        (outline_p[:, 1] > p1[1])]
        backp2 = outline_p[np.isclose(outline_p[:,1], slope_perp_p2*outline_p[:,0] + b_perp_p2,atol=1) & 
                        (outline_p[:, 0] > p1[0]) & 
                        (outline_p[:, 1] > p1[1])]
        
    if len(backp1)>0 and len(backp2)>0:
        backp1 = backp1[0]
        backp2 = backp2[0]
    else: 
        return False

    mid_back = midpoint(backp1,backp2)
    slope_perp_back, perp_b = perpendicular(backp1,backp2,mid_back)

    if left:
        straight = outline_p[np.isclose(outline_p[:,1], slope_perp_back*outline_p[:,0] + perp_b,atol=1) & 
                        (outline_p[:, 0] < mid_back[0]) & 
                        (outline_p[:, 1] > mid_back[1])]
    else:
        straight = outline_p[np.isclose(outline_p[:,1], slope_perp_back*outline_p[:,0] + perp_b,atol=1) & 
                        (outline_p[:, 0] > mid_back[0]) & 
                        (outline_p[:, 1] > mid_back[1])]
    if len(straight)>0:
        return False
    else:
        return True
    
pose = YOLO('yolov8n-pose.pt')
pose.to('cuda')
outline = YOLO('yolov8n-seg.pt')
outline.to('cuda')
cap = cv2.VideoCapture('Squat.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output2.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # get pose of biggest "box" of person in a frame
    # gets the outline of the biggest "box" of person in a frame
    annotated_pose, largest_idx_pose, person_detected = predict(pose,frame)
    annotated_outline, largest_idx_outline, person_detected = predict(outline,frame)

    if person_detected:
        pose_coords = annotated_pose.keypoints[largest_idx_pose].data.cpu().numpy() # get the pose coordinates of biggest box 
        outline_coords = annotated_outline.masks.xy[largest_idx_outline] # get the outline coordinates of biggest box

        h, w, c= frame.shape  
        canvas = np.zeros((h,w,c),dtype=np.uint8) # make black and white canvas
        for point in pose_coords:
            for x,y,conf in point:
                cv2.circle(canvas,(int(x),int(y)),5,(255,255,255),-1)
        for a,b in outline_coords:
            cv2.circle(canvas,(int(a),int(b)),5,(255,255,255),-1)
        # cv2.imshow('combined',canvas)
        # print(straight_back(pose_coords,outline_coords))
        out.write(canvas)

    #cv2.imshow('pose',annotated_pose.plot())
    #cv2.imshow('outline',annotated_outline.plot())
    # termination --> press q on keyboard
    '''
    if 0xFF == ord('q'):  
        cv2.destroyAllWindows
        out.release
        break
    '''