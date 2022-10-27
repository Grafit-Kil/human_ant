from re import X
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




def calculate_angle(a,b,c):
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Background lines
def drawBackground(frm):
    rng = 80
    for x in range(6):
        cv2.line(frm,(0,rng*x),(640,rng*x),(170,170,170),1)
    for x in range(8):
        cv2.line(frm,(rng*x,0),(rng*x,480),(100,100,100),1)


# Camera set for 640x480 resolution
# If different resize
# 
cap = cv2.VideoCapture(0)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
 
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)
    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        ret2, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
        
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass
        
        print(landmarks)
        
        # Angles 
        # LEFT ELBOW
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        left = calculate_angle(left_shoulder,left_elbow,left_wrist)
        
        # RIGHT ELBOW
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        right = calculate_angle(right_shoulder,right_elbow,right_wrist)
        cv2.rectangle(thresh1, (0,0), (640,480), (0,0,0), -1)
        drawBackground(thresh1)
        mp_drawing.draw_landmarks(thresh1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.putText(thresh1, "SOL DIRSEK = " + str(left), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(thresh1, "SAG DIRSEK = " + str(right), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Main', image)

        cv2.imshow('thr', thresh1)

        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()