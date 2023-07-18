from realSenseWrapper import RealsenseManager

import cv2
import numpy as np

import open3d as o3d



def load_stereo_coefficients(number1, number2):
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FileStorage_READ)
    leftMapX = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_x").mat()
    leftMapY = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_y").mat()
    rightMapX = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_x").mat()
    rightMapY = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_y").mat()
    Q = cv_file.getNode(f"M{number1}-{number2}Q").mat()
    cv_file.release()
    return leftMapX, leftMapY, rightMapX, rightMapY, Q


def nothing(x):
    pass



def main():

    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)
    
    cv2.createTrackbar('numDisparities','disp',1,17,nothing)
    cv2.createTrackbar('blockSize','disp',5,50,nothing)
    cv2.createTrackbar('preFilterType','disp',1,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
    cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
    cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
    cv2.createTrackbar('speckleRange','disp',0,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
    cv2.createTrackbar('minDisparity','disp',5,25,nothing)


    rsm = RealsenseManager()
    rsm.enableDevices()
    m01leftMapx, m01leftMapY, m01RightMapx, m01RightMapy, m01Q = load_stereo_coefficients(1, 0)
    m12leftMapx, m12leftMapY, m12RightMapx, m12RightMapy, m12Q = load_stereo_coefficients(0, 2)
    m23leftMapx, m23leftMapY, m23RightMapx, m23RightMapy, m23Q = load_stereo_coefficients(2, 3)

    current = 0

    stereo = cv2.StereoBM_create()

    while True:
        frames = rsm.get_frames()

        f0 =  cv2.remap(frames[1], m01leftMapY, m01leftMapx, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        f1 =  cv2.remap(frames[0], m01RightMapx, m01RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f2 =  cv2.remap(frames[0], m12leftMapx, m12leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f3 =  cv2.remap(frames[2], m12RightMapx, m12RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f4 =  cv2.remap(frames[2], m23leftMapx, m23leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        f5 =  cv2.remap(frames[3], m23RightMapx, m23RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        rectified = [f0, f1, f2, f3, f4, f5]

        Left_nice = f0
        Right_nice = f1

        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
     

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
            # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
    
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        point_cloud = cv2.reprojectImageTo3D(disparity, m01Q)
        point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])
        point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]


        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcl])

    
        # Displaying the disparity map
        cv2.imshow("disp",disparity)
    
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break




if __name__ == "__main__":
    main()