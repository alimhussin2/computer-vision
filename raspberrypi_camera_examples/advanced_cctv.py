#!/usr/bin/python3
# DESCRIPTIONS: Record video using OpenCV

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import os
import datetime
import time
import imutils

class SmartCctv():

    def __init__(self):
        print('[INFO]: Initializing camera...')
        self.width = 1280
        self.height = 720
        self.record_time = 30 # record for 10 seconds
        self.camera = PiCamera()
        self.camera.resolution = (self.width, self.height)
        self.camera.framerate = 24
        self.rawCapture = PiRGBArray(self.camera, size=(self.width, self.height))
        time.sleep(2)
        print('[INFO]: Initializing camera completed')

    def get_capture_time():
        return datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S%p")

    def mode_track(self):
        print('[INFO]: Mode:  Tracking')
        capture_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S%p")
        vid_file = "mode_track_" + capture_time + ".avi"
        out = cv2.VideoWriter(vid_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self.camera.framerate, (self.width, self.height))
        record_vid = cv2.VideoCapture(0)
        avg = None
        start = time.time()
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            img_frame = frame.array
            gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)
            if avg is None:
                avg = gray.copy().astype("float")
                self.rawCapture.truncate(0)
                continue
            cv2.accumulateWeighted(gray, avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            thresh = cv2.threshold(frameDelta, 25,255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            print('Recording...')
            for c in cnts:
                if cv2.contourArea(c) < 50:
                    #print('[INFO]: Idle...')
                    continue
                print('[INFO]: Recording...')
                cv2.putText(img_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(img_frame)
            cv2.waitKey(1)
            self.rawCapture.truncate(0)
            if time.time() - start > self.record_time:
                print('[INFO]: Recording ended')
                break
        out.release()
        record_vid.release()

    def mode_headless(self):
        # record video using picamera library
        print('[INFO]: Mode: Headless')
        capture_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S%p")
        vid_file = 'cctv_vid_' + capture_time + '.h264'

        print('[INFO]: Start recording...')
        self.camera.start_recording(vid_file)
        time.sleep(self.record_time)
        print('[INFO]: Recording ended')
        print('[INFO]: Saved video as %s' % (vid_file))
        self.camera.stop_recording()

    def mode_snap_photo(self):
        print('[INFO]: Mode: Snap photo')
        w = 2560
        h = 1960
        #self.camera.resolution = (w, h)
        #time.sleep(2)
        capture_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S%p")
        photo_file = 'capture_' + capture_time + '.jpg'
        self.camera.capture(photo_file, resize(w,h))

    def main(self):
        print('Advanced CCTV')
        while True:
            print('CCTV Mode')
            print('Recording video')
            print('  [1] Headless')
            print('  [2] Detect motion')
            print('  [3] Continuous detection')
            print('Capture image')
            print('  [4] Snap Photo')
            print('Live Streaming')
            print('  [5] Live video streaming')
            print('  [q] Exit')
            key = input('Select mode >> ')
            if key == '1':
                self.mode_headless()
            elif key == '2':
                self.mode_track()
            elif key == '3':
                while True:
                    f = cv2.waitKey(50)
                    if f == 27: # ESC key
                        break
                    else:
                        self.mode_track()
            elif key == '4':
                self.mode_snap_photo()
            elif key == 'q':
                break
            else:
                print('[ERROR]: Invalid input')

cctv = SmartCctv()
cctv.main()

