import cv2
import numpy as np 
import mediapipe as mp 
import time 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

#Curl Counter variable
counter = 0
stage = None


## set mediapip instance: 
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:  ##confidence means how accurat the detection is but                                                                                             to high could mean no detection...
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolor Image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Make Detection
        results = pose.process(image)

        # Recolor Image from RGB back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinats
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

    
        except:
            pass


        #Render Detections --> Showing landmarks and dots
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=5), #Giving the DOTS another color
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  #Giving the LINES another color 
                                )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    