from realSenseWrapper import RealsenseManager

import cv2
import numpy as np


def load_stereo_coefficients(number1, number2):
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FileStorage_READ)
    leftMapX = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_x").mat()
    leftMapY = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_y").mat()
    rightMapX = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_x").mat()
    rightMapY = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_y").mat()
    cv_file.release()
    return leftMapX, leftMapY, rightMapX, rightMapY


def main():
    rsm = RealsenseManager()
    rsm.enableDevices()
    m01leftMapx, m01leftMapY, m01RightMapx, m01RightMapy = load_stereo_coefficients(0, 1)

    while True:
        frames = rsm.get_frames()

        f0 = cv2.remap(frames[0], m01leftMapx, m01leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f1 =  cv2.remap(frames[1], m01RightMapx, m01RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        cv2.imshow("img", frames[0])
        cv2.imshow("rectified", f0)
        k = cv2.waitKey(1)

        if(k==ord('q')):break





if __name__ == "__main__":
    main()