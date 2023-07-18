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
    m12leftMapx, m12leftMapY, m12RightMapx, m12RightMapy = load_stereo_coefficients(1, 2)
    m23leftMapx, m23leftMapY, m23RightMapx, m23RightMapy = load_stereo_coefficients(2, 3)

    while True:
        frames = rsm.get_frames()

        f0 =  cv2.remap(frames[0], m01leftMapx, m01leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f1 =  cv2.remap(frames[1], m01RightMapx, m01RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f2 =  cv2.remap(frames[1], m12leftMapx, m12leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f3 =  cv2.remap(frames[2], m12RightMapx, m12RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f4 =  cv2.remap(frames[2], m23leftMapx, m23leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f5 =  cv2.remap(frames[3], m23RightMapx, m23RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        cv2.imshow("img", frames[3])
        cv2.imshow("rectified", f5)
        k = cv2.waitKey(1)

        if(k==ord('q')):break





if __name__ == "__main__":
    main()