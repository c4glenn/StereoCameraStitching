import pyrealsense2 as rs
import numpy as np
import cv2
from realSenseWrapper import RealsenseManager
import datetime

INTRINSIC = True

filename = f"Calibration/images/{'Intrinsic' if INTRINSIC else 'StereoPairs'}/"


def main():
    rsm = RealsenseManager(None, False, False)
    rsm.enableDevices()

    while True:
        frames = rsm.get_frames()
        frames = [cv2.resize(frame, dsize=(480,270)) for frame in frames]
        frames = [frames[0], frames[1], frames[2], frames[3]]
        cv2.imshow("img", np.hstack(frames))
        k = cv2.waitKey(1)
        if(k==ord("q")): break
        if(k==ord("s")):
            time = datetime.datetime.now()
            for i, frame in enumerate(frames):
                cv2.imwrite(filename+f"{i}/{time.timestamp()}.jpg", frame)




if __name__ == "__main__":
    main()