import cv2
import numpy as np
from math import sqrt
from math import acos
from math import pi
from math import degrees
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
    if A==0 or B == 0:
        return 0
    cosine_law = ((C**2 - A**2 - B**2) / (-2*A*B))
    cos_theta = max(-1,min(1,cosine_law))
    angle = abs(degrees(acos(cos_theta)))
    return angle 

# returns the slope and y-intercept given by 2 points in y = mx + b format 
def line(p1, p2):
    if p2[0]-p1[0]==0:
        slope = 100
    else:
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
def exercise_detection(pose_coords):
    a_12_14_16 = angle(pose_coords[0][12],pose_coords[0][14],pose_coords[0][16])
    a_11_13_15 = angle(pose_coords[0][11],pose_coords[0][13],pose_coords[0][15])
    a_6_8_10 = angle(pose_coords[0][6],pose_coords[0][8],pose_coords[0][10])
    a_5_7_9 = angle(pose_coords[0][5],pose_coords[0][7],pose_coords[0][9])
    a_6_12_16 = angle(pose_coords[0][6],pose_coords[0][12],pose_coords[0][16])
    a_5_11_15 = angle(pose_coords[0][5],pose_coords[0][11],pose_coords[0][15])
    a_8_6_12 = angle(pose_coords[0][8],pose_coords[0][6],pose_coords[0][12])
    a_7_5_11 = angle(pose_coords[0][7],pose_coords[0][5],pose_coords[0][11])
    a_10_6_12 = angle(pose_coords[0][10],pose_coords[0][6],pose_coords[0][12])
    a_9_5_11 = angle(pose_coords[0][9],pose_coords[0][5],pose_coords[0][11])
    a_6_12_14 = angle(pose_coords[0][6],pose_coords[0][12],pose_coords[0][14])
    a_5_11_13 = angle(pose_coords[0][5],pose_coords[0][11],pose_coords[0][13])
    a_3_5_11 = angle(pose_coords[0][3],pose_coords[0][5],pose_coords[0][11])
    a_4_6_12 = angle(pose_coords[0][4],pose_coords[0][6],pose_coords[0][12])
    a_4_12_16 = angle(pose_coords[0][4],pose_coords[0][12],pose_coords[0][16])
    a_3_11_15 = angle(pose_coords[0][3],pose_coords[0][11],pose_coords[0][15])


    if ((160<=a_3_5_11<=180 or 160<=a_4_6_12<=180) and
          (160<=a_6_8_10<=180 or 160<=a_5_7_9<=180) and
          (120<=a_12_14_16<=180 or 120<=a_11_13_15<=180) and
          (90<=a_4_12_16 or 90<=a_3_11_15)):
        return "deadlift"
    
    elif ((160<=a_6_12_16<=180 or 160<=a_5_11_15<=180) and
           (0<=a_5_7_9<=90 or 0<=a_6_8_10<=90) and
           (0<=a_8_6_12<=90 or 0<=a_7_5_11<=90) and
           (160<=a_12_14_16<=180 or 160<=a_11_13_15<=180)):
        return "squat"
    elif ((120<=a_10_6_12<=180 or 120<=a_9_5_11<=180) and
          (a_12_14_16<=120 or a_11_13_15<=120) and 
          (160<=a_6_8_10<=180 or 160<=a_5_7_9<=180)):
        return "lat"
    elif ((130<=a_10_6_12<=180 or 130<=a_9_5_11<=180) and
          ((0<=a_12_14_16<=90 or 0<=a_11_13_15<=90) or 
           (160<=a_6_12_16<=180 or 160<=a_5_11_15<=180))):
        return "pul"
    elif ((160<=a_5_7_9<=180 or 160<=a_6_8_10<=180) and
          (120<=a_6_12_14<=180 or 120<=a_5_11_13<=180) and 
          (0<=a_12_14_16<=120 or 0<=a_11_13_15<=120)):
        return "dip"
    else:
        return "No exercise recognized"

def lat_form(frame, pose_coords):
    """Modified to return feedback instead of writing directly to frame"""
    # Elbow angles
    a_6_8_10 = angle(pose_coords[0][6], pose_coords[0][8], pose_coords[0][10])
    a_5_7_9 = angle(pose_coords[0][5], pose_coords[0][7], pose_coords[0][9])
    elbow_angle = min(a_6_8_10, a_5_7_9)

    a_6_12_14 = angle(pose_coords[0][6], pose_coords[0][12], pose_coords[0][14])
    a_5_11_13 = angle(pose_coords[0][5], pose_coords[0][11], pose_coords[0][13])
    lean_angle = max(a_6_12_14, a_5_11_13)

    feedback = []

    if 120 < elbow_angle < 150:
        feedback.append(('Extension: Full Extension at Top of Rep!', (0, 0, 255)))
    else:
        feedback.append(('Extension: Good!', (0, 255, 0)))

    if 0 < lean_angle < 135:
        feedback.append(('Lean: Good!', (0, 255, 0)))
    else:
        feedback.append(('Lean: Too Far Back!', (0, 0, 255)))

    if 90 <= elbow_angle <= 120:
        feedback.append(('Pull: Bring Bar to Chest!', (0, 0, 255)))
    else:
        feedback.append(('Pull: Good!', (0, 255, 0)))

    return feedback

def pul_form(frame, pose_coords):
    a_6_8_10 = angle(pose_coords[0][6], pose_coords[0][8], pose_coords[0][10])
    a_5_7_9 = angle(pose_coords[0][5], pose_coords[0][7], pose_coords[0][9])
    elbow_angle = min(a_6_8_10, a_5_7_9)

    feedback = []

    if 100 <= elbow_angle <= 150:
        feedback.append(("Extension: Full Extension at Top of Rep!", (0, 0, 255)))
    else:
        feedback.append(("Extension: Good!", (0, 255, 0)))

    if 60 <= elbow_angle <= 90:
        feedback.append(("Pull: Bring Chin to Bar!", (0, 0, 255)))
    else:
        feedback.append(("Pull: Good!", (0, 255, 0)))

    return feedback

def dips_form(frame, pose_coords):
    a_6_8_10 = angle(pose_coords[0][6], pose_coords[0][8], pose_coords[0][10])
    a_5_7_9 = angle(pose_coords[0][5], pose_coords[0][7], pose_coords[0][9])
    elbow_angle = min(a_6_8_10, a_5_7_9)

    feedback = []

    if 120 <= elbow_angle <= 150:
        feedback.append(("Extension: Full Extension at Top of Rep!", (0, 0, 255)))
    else:
        feedback.append(("Extension: Good!", (0, 255, 0)))

    if 90 <= elbow_angle < 120:
        feedback.append(("Push: Bring Chest Lower!", (0, 0, 255)))
    else:
        feedback.append(("Push: Good!", (0, 255, 0)))

    return feedback

def deadlift_form(frame, pose_coords, outline_coords):
    feedback = []

    a_6_12_16 = angle(pose_coords[0][6], pose_coords[0][12], pose_coords[0][16])
    a_5_11_15 = angle(pose_coords[0][5], pose_coords[0][11], pose_coords[0][15])
    straight = max(a_6_12_16, a_5_11_15)

    a_10_14_16 = angle(pose_coords[0][10], pose_coords[0][14], pose_coords[0][16])
    a_9_13_15 = angle(pose_coords[0][9], pose_coords[0][13], pose_coords[0][15])
    legs = max(a_9_13_15, a_10_14_16)

    if straight_back(pose_coords, outline_coords):
        feedback.append(("Back: Good!", (0, 255, 0)))
    else:
        feedback.append(("Back: Keep Back Straight!", (0, 0, 255)))

    if 150 <= straight <= 170:
        feedback.append(("Pull: Lock Out Deadlift!", (0, 0, 255)))
    else:
        feedback.append(("Pull: Good!", (0, 255, 0)))

    if legs >= 90:
        feedback.append(("End of lift: All the Way Down!", (0, 0, 255)))
    else:
        feedback.append(("End of lift: Good!", (0, 255, 0)))

    return feedback

def squat_form(frame, pose_coords, outline_coords):
    feedback = []

    a_12_14_16 = angle(pose_coords[0][12], pose_coords[0][14], pose_coords[0][16])
    a_11_13_15 = angle(pose_coords[0][11], pose_coords[0][13], pose_coords[0][15])
    legs = max(a_11_13_15, a_12_14_16)

    if straight_back(pose_coords, outline_coords):
        feedback.append(("Back: Good!", (0, 255, 0)))
    else:
        feedback.append(("Back: Keep Back Straight!", (0, 0, 255)))

    if legs >= 90:
        feedback.append(("Legs: All the Way Down!", (0, 0, 255)))
    else:
        feedback.append(("Legs: Good!", (0, 255, 0)))

    return feedback
