#!/usr/bin/python3

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils

haar_file = "../data/haarcascades/haarcascade_frontalface_default.xml"
#haar_file = "/home/pi/opencv/data/haarcascades/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(haar_file)

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(640,480))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect the faces
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    #print("Found %s faces!" % (format(len(faces))))

    # draw rectangle around the face
    for(x,y,w,h) in faces:
        #print("%s" % str(int((w+h)/3)))
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        #cv2.imwrite("/home/pi/image.jpg", image)
    cv2.imshow("LiveStream", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break
