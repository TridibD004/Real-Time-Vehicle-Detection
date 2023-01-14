"""
 Real Time Vehicle Detection
 @author: Tridib Dalui
"""

import cv2
import time

# To create our body classifier
car_classifier=cv2.CascadeClassifier('isCar.xml')

#initiate capturing video from file
cap=cv2.VideoCapture('video1.mp4')

#loop untill video ends
while cap.isOpened():
    time.sleep(.05)

    #read first frame
    r, frame= cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #pass frame to car classifier
    cars=car_classifier.detectMultiScale(gray, 1.4,2)

    # Creating boundary boxs
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('Cars',frame)

    #break if Enter key is pressed
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()