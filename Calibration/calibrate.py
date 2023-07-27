import cv2
import numpy as np
import os
import glob


CHECKERBOARD = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
criteria_stereo = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
	
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC


def process_frame(fname, objpoints, imgpoints, objp) -> tuple[list, list]:
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
        cv2.waitKey(10)
    
    return objpoints, imgpoints



def calibrate_single_camera(number, intrinsic:bool=True):
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    filename = f'Calibration/images/{"Intrinsic" if intrinsic else "StereoPairs"}/{number}/*'

    images = glob.glob(filename)
    last_ret = 0

    h,w = cv2.imread(images[0]).shape[:2]

    for fname in images:
        objpoints, imgpoints = process_frame(fname, objpoints, imgpoints, objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h,w), None, None)

    new_mtxl, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    print(f"rsme: {ret}")

    return (dist, new_mtxl)


def calibrate_pair(number1, number2):
    distL, new_mtxL = calibrate_single_camera(number1, True)
    distR, new_mtxR = calibrate_single_camera(number2, True)

    obj_pts = []

    imgPointsL = []
    imgPointsR = []

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    filenameL = f'Calibration/images/StereoPairs/{number1}/*'
    filenameR = f'Calibration/images/StereoPairs/{number2}/*'

    imagesL = glob.glob(filenameL)
    imagesR = glob.glob(filenameR)

    for fnameL, fnameR in zip(imagesL, imagesR):
        imgL = cv2.imread(fnameL)
        imgR = cv2.imread(fnameR)
        imgL_gray = cv2.imread(fnameL,0)
        imgR_gray = cv2.imread(fnameR,0)

        outputL = imgL.copy()
        outputR = imgR.copy()

        retR, cornersR =  cv2.findChessboardCorners(outputR,CHECKERBOARD,None)
        retL, cornersL = cv2.findChessboardCorners(outputL,CHECKERBOARD,None)

        if retR and retL:
            obj_pts.append(objp)
            cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
            cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
            cv2.drawChessboardCorners(outputR,CHECKERBOARD,cornersR,retR)
            cv2.drawChessboardCorners(outputL,CHECKERBOARD,cornersL,retL)
            cv2.imshow('img',np.hstack([outputL, outputR]))
            cv2.waitKey(10)
        
            imgPointsL.append(cornersL)
            imgPointsR.append(cornersR)

    h,w = cv2.imread(imagesL[0]).shape[:2]

    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, imgPointsL, imgPointsR, new_mtxL, distL, new_mtxR, distR, (w, h), criteria=criteria_stereo, flags=flags)

    print(f"stereo pair {number1} - {number2}")
    print(f"stero rsme:{retS}")
    print(f"ROT: \n{Rot}")
    print(f"Trns: \n", Trns)
    rectify_scale= 2.54
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::-1],Rot, Trns, rectify_scale,(0,0), flags=cv2.CALIB_ZERO_DISPARITY)

    Left_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                            imgL_gray.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map= cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                            imgR_gray.shape[::-1], cv2.CV_16SC2)
    
    print("Saving paraeters ......")
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_APPEND)
    cv_file.write(f"M{number1}-{number2}L_Stereo_Map_x",Left_Stereo_Map[0])
    cv_file.write(f"M{number1}-{number2}L_Stereo_Map_y",Left_Stereo_Map[1])
    cv_file.write(f"M{number1}-{number2}R_Stereo_Map_x",Right_Stereo_Map[0])
    cv_file.write(f"M{number1}-{number2}R_Stereo_Map_y",Right_Stereo_Map[1])
    cv_file.write(f"M{number1}-{number2}Q",Q)
    cv_file.release()

if __name__ == "__main__":
    calibrate_pair(1,0)
    calibrate_pair(0,2)
    calibrate_pair(2,3)