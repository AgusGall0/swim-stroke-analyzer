import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision   


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.IMAGE)
with PoseLandmarker.create_from_options(options) as landmarker:



    
cap = cv.VideoCapture('ejemplo.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv.imshow('frame', frame)
    if cv.waitKey(24) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()