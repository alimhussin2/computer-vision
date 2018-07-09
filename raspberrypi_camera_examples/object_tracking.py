#!/usr/bin/python3

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import sys

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(640,480))
time.sleep(0.1)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        ttacker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img_frame = frame.array
    bbox = (200,50,100,250)

    # uncomment line below to set another object to track
    #bbox = cv2.selectROI(img_frame, False)
    ok = tracker.init(img_frame, bbox)
    timer = cv2.getTickCount()
    # update tracker
    ok, bbox = tracker.update(img_frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # draw box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img_frame, p1, p2, (255,0,0), 2 ,1)
    else:
        cv2.putText(img_frame, "Tracking failure detected", (100,80),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(00,255),2)
        cv2.putText(img_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    cv2.imshow("LiveStream", img_frame)

    # quit the camera
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break

