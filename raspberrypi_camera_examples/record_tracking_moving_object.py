#!/usr/bin/python3

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils
import datetime

capture_time = datetime.datetime.now().strftime("%Y_%m_%d%_H%M%S%p")
vid_file = "video_object_track_" + capture_time + ".avi"
width = 640
height = 480
frameRate = 24
camera = PiCamera()
camera.resolution = (width,height)
camera.framerate = frameRate
rawCapture = PiRGBArray(camera, size=(width,height))
record_vid = cv2.VideoCapture(0)
out = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (width, height))
avg = None
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img_frame = frame.array
    #img_frame = imutils.resize(img_frame, width=500)
    gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    if avg is None:
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 25,255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        if cv2.contourArea(c) < 3000:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img_frame, (x,y), (x+w, y+h), (0,255,0),2)
        cv2.putText(img_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("LiveStream", img_frame)
    cv2.imshow("thresh", thresh)
    out.write(img_frame)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break
out.release()
record_vid.release()
cv2.destroyAllWindows()
