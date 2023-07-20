from realSenseWrapper import RealsenseManager
from V0PointCloudGen import Visualizer

import cv2
import numpy as np

import open3d as o3d
import open3d.visualization as vis


global count

count = 0

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


def setup_openCV(name):
    cv2.namedWindow('disp'+name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp'+name,600,600)
    
    cv2.createTrackbar('numDisparities','disp'+name,5,17,nothing)
    cv2.createTrackbar('blockSize','disp'+name,7,50,nothing)
    cv2.createTrackbar('preFilterType','disp'+name,0,1,nothing)
    cv2.createTrackbar('preFilterSize','disp'+name,3,25,nothing)
    cv2.createTrackbar('preFilterCap','disp'+name,62,62,nothing)
    cv2.createTrackbar('textureThreshold','disp'+name,45,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp'+name,0,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp'+name,16,25,nothing)
    cv2.createTrackbar('minDisparity','disp'+name,7,25,nothing)


def updateOpenCV(stereo, left_nice, right_nice, name):
        numDisparities = cv2.getTrackbarPos('numDisparities','disp'+name)*16
        blockSize = cv2.getTrackbarPos('blockSize','disp'+name)*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp'+name)
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp'+name)*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp'+name)
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp'+name)
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp'+name)*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp'+name)
        minDisparity = cv2.getTrackbarPos('minDisparity','disp'+name)
     

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        return stereo, calculate_disparity(stereo, left_nice, right_nice, minDisparity, numDisparities)


def calculate_disparity(stereo, Left_nice, Right_nice, minDisparity, numDisparities):
            # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
    
        # Converting to float32 
        disparity = disparity.astype(np.float32)
    
        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        return disparity


def pointCloudFromDisparity(disparity, q):
    global count
    h, w = disparity.shape[:2]
    f=.8*w
    Q = np.float32([[1, 0, 0,      0],
                    [0,-1, 0,      0],
                    [0, 0, f*0.05, 0],
                    [0, 0, 1,      0]])
    pcl = o3d.geometry.PointCloud()
    disparity.astype(np.float32)

    mask = disparity > disparity.min()

    point_cloud = cv2.reprojectImageTo3D(disparity, q)
    cv2.imshow("img", point_cloud)
    point_cloud = point_cloud.reshape(-1, 3)
    point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]

    
    if count == 10:
        np.save("pointcloudFromV1", np.asarray(point_cloud))
        print("saved")
    count += 1 
   
    #point_cloud = point_cloud[mask]

    print(point_cloud, len(point_cloud), len(point_cloud[0]))

    pcl.points = o3d.utility.Vector3dVector(point_cloud)
    
    print(np.asarray(pcl.points), len(pcl.points))
    pcl.paint_uniform_color([1, 0.706, 0])

    return pcl

def getRectifiedFrames(rsm, M10, M02, M23):
    frames = rsm.get_frames()
    m01leftMapx, m01leftMapY, m01RightMapx, m01RightMapy, m01Q = M10
    m12leftMapx, m12leftMapY, m12RightMapx, m12RightMapy, m12Q = M02
    m23leftMapx, m23leftMapY, m23RightMapx, m23RightMapy, m23Q = M23

    f0 =  cv2.remap(frames[1], m01leftMapY, m01leftMapx, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    f1 =  cv2.remap(frames[0], m01RightMapx, m01RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    f2 =  cv2.remap(frames[0], m12leftMapx, m12leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    f3 =  cv2.remap(frames[2], m12RightMapx, m12RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    f4 =  cv2.remap(frames[2], m23leftMapx, m23leftMapY, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    f5 =  cv2.remap(frames[3], m23RightMapx, m23RightMapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    rectified = [f0, f1, f2, f3, f4, f5]
    return rectified

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    rsm = RealsenseManager()
    rsm.enableDevices()

    M10 = load_stereo_coefficients(1, 0)
    M02 = load_stereo_coefficients(0, 2)
    M23 = load_stereo_coefficients(2, 3)        # Displaying the disparity map


    stereo = cv2.StereoBM_create()
    setup_openCV("M10")

    frames = getRectifiedFrames(rsm, M10, M02, M23)
    stereo, disp = updateOpenCV(stereo, frames[0], frames[1], "M10")

    geometry = pointCloudFromDisparity(disp, M10[4])

    v = Visualizer()
    v.do_once(geometry)

    while True:
        frames = getRectifiedFrames(rsm, M10, M02, M23)
        stereo, disp = updateOpenCV(stereo, frames[0], frames[1], "M10")
        points = cv2.reprojectImageTo3D(disp, M10[4])
        colors = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

        mask = disp > disp.min()

        #write_ply(out_fn, out_points, out_colors)

        point_cloud = pointCloudFromDisparity(disp, M10[4]).points
        
        geometry.points = point_cloud
        cv2.imshow("dispM10",disp)

        v.do_each_loop(geometry)
        #Close window using esc key
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()