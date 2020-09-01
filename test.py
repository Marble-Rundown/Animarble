import cv2

cap = cv2.VideoCapture('BobTopDown_mocap.mp4')
while cap.isOpened():
    s, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('test', gray)