import cv2
from realSenseWrapper import RealsenseManager
import datetime
import os

rsm = RealsenseManager(color=True)
rsm.enableDevices()

while True:
    time = datetime.datetime.now().timestamp()
    frames = rsm.get_frames()


    names = ["frame0", "frame1", "frame2", "frame3", "frame4", "frame5"]

    for i, frame in enumerate(frames):
        cv2.imwrite(f"VideoForTests/{names[i]}/{time}.jpg", frame)
