import cv2
import mediapipe as mp
import numpy as np
import math as m
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def get_distance(x1,y1,x2,y2):
	x_dist = (x1 - x2)
	y_dist = (y1 - y2)
	return m.sqrt(x_dist * x_dist + y_dist * y_dist)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_pose.Pose( 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if not results.pose_landmarks:
    	continue
    	
    left_hand = False
    right_hand = False
    leg = False

    #left hand
    lwx=results.pose_landmarks.landmark[15].x*image.shape[0]
    lwy=results.pose_landmarks.landmark[15].y*image.shape[1]
    lhx=results.pose_landmarks.landmark[23].x*image.shape[0]
    lhy=results.pose_landmarks.landmark[23].y*image.shape[1]
    left_hand_distance = get_distance(lwx,lwy,lhx,lhy)
    if (left_hand_distance/4) >= 85:
    	left_hand = True
    
    # #right hand
    rwx=results.pose_landmarks.landmark[16].x*image.shape[0]
    rwy=results.pose_landmarks.landmark[16].y*image.shape[1]
    rhx=results.pose_landmarks.landmark[24].x*image.shape[0]
    rhy=results.pose_landmarks.landmark[24].y*image.shape[1]
    right_hand_distance = get_distance(rwx,rwy,rhx,rhy)
    if (right_hand_distance/4) >= 85:
    	right_hand = True

    # #leg distance
    lkx=results.pose_landmarks.landmark[27].x*image.shape[0]
    lky=results.pose_landmarks.landmark[27].y*image.shape[1]
    rkx=results.pose_landmarks.landmark[28].x*image.shape[0]
    rky=results.pose_landmarks.landmark[28].y*image.shape[1]
    leg_distance = get_distance(lkx,lky,rkx,rky)
    if leg_distance >= 100:
    	leg=True

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    clear_body = True
    for landmark in results.pose_landmarks.landmark:
    	if landmark.visibility < 0.50:
    		clear_body = False

    if clear_body == True:
    	if left_hand == False and right_hand == False:
    		msg = "Rise your hand."
    	
    	if left_hand == False and right_hand == True:
    		msg = "Rise your left side hand."
    	
    	if left_hand == True and right_hand == False:
    		msg = "Rise your right side hand."

    	if left_hand == True and right_hand == True and leg == False:
    		msg = "Set your legs postion."

    	if left_hand == True and right_hand == True and leg == True:
    		msg = "Perfect Dude!!"
    	
    else:
    	msg = "Cannot see full body"
    

    if msg:
    	image = cv2.putText(cv2.flip(image,1), msg, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)	
    
    cv2.imshow('MediaPipe Pose',image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()