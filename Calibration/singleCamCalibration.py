import cv2
import glob
import numpy as np


def getImages(filename) -> list[cv2.Mat]:
    images_folder = filename
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)
    return images

def calibrateSingleCam(images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER ,30, 0.001)

    rows = 5
    columns = 7
    world_scaling = 2.54 # ish CM grid squares

    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = images[0].shape[1]
    height = images[0].shape[0]
    
    imgpoints = []
    objpoints = []

    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if(ret==True):
            conv_size = (11, 11)

            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.imshow('IMG', frame)
            k = cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    #print('camera matrix:\n', mtx)
    #print('distortion coeffs:', dist)
    #print('Rs:\n', rvecs)
    #print('Ts:\n', tvecs)

    return mtx, dist
    


if __name__ == "__main__":
    L1 = getImages("Calibration/images/927522071127-L/*")
    print("L1 ***************************************")
    mtx, dist = calibrateSingleCam(L1)
    R1 = getImages("Calibration/images/927522071127-R/*")
    print("R1 ***************************************")
    mtx, dist = calibrateSingleCam(R1)
    L2 = getImages("Calibration/images/927522073022-L/*")
    print("L2 ***************************************")
    mtx, dist = calibrateSingleCam(L2)
    R2 = getImages("Calibration/images/927522073022-R/*")
    print("R2 ***************************************")
    mtx, dist = calibrateSingleCam(R2)