import cv2
import numpy as np
import os
import glob


CHECKERBOARD = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
criteria_stereo = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
	
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

def calibrate_single_camera(number, intrinsic:bool=True):
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    filename = f'Calibration/images/{"Intrinsic" if intrinsic else "Extrinsic"}/{number}/*'

    images = glob.glob(filename)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
            cv2.imshow('img',img)
            cv2.waitKey(500)    

    cv2.destroyAllWindows()
    h,w = img.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    new_mtxl, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    print(f"rsme: {ret}")

    return (mtx, dist, new_mtxl, roi)


def calibrate_pair(number1, number2):
    mtxL, distL, new_mtxL, roiL = calibrate_single_camera(number1, False)
    mtxR, distR, new_mtxR, roiR = calibrate_single_camera(number2, False)



if __name__ == "__main__":
    for i in range(4):
        print(i)
        calibrate_single_camera(i)