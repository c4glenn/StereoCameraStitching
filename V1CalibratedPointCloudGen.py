from realSenseWrapper import RealsenseManager
from V0PointCloudGen import Visualizer, Pose, CAMERA_POSES, TRANSLATIONAL_SCALE
from tqdm import tqdm


import cv2
import numpy as np
from numpy import cos, sin

import open3d as o3d
import open3d.visualization as vis


global count


count = 0

def create_extrinsic(pose: Pose) -> np.array:
        ca, a, sa = cos(pose.rX), pose.rX, sin(pose.rX)
        cb, b, sb = cos(pose.rY), pose.rY, sin(pose.rY)
        cc, c, sc = cos(pose.rZ), pose.rZ, sin(pose.rZ)
        x = pose.x * TRANSLATIONAL_SCALE
        y = pose.y * TRANSLATIONAL_SCALE
        z = pose.z * TRANSLATIONAL_SCALE
        return np.array([
            [cb*cc, -cb*sc, sb, x*cb*cc - y*cb*sc],
            [sa*sb*cc + ca*sc, ca*cc-sa*sb*sc, -sa*cb, x*(sa*sb*cc+ca*sc) + y*(ca*cc-sa*sb*sc) - z*(sa*cb)],
            [sa*sc-ca*sb*cc, ca*sb*sc+sa*cc, ca*cb, x*(sa*sc-ca*sb*cc)+y*(ca*sb*sc+sa*cc) + z*(ca*cb)],
            [0, 0, 0, 1]
        ])

def load_stereo_coefficients(number1, number2):
    cv_file = cv2.FileStorage("CalibrationParameters.xml", cv2.FileStorage_READ)
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
    
    cv2.createTrackbar('numDisparities','disp'+name,7,17,nothing)
    cv2.createTrackbar('blockSize','disp'+name,3,50,nothing)
    cv2.createTrackbar('preFilterCap','disp'+name,0,62,nothing)
    cv2.createTrackbar('speckleWindowSize','disp'+name,0,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp'+name,25,25,nothing)
    cv2.createTrackbar('minDisparity','disp'+name,19,25,nothing)



def updateOpenCV(stereo, left_nice, right_nice, name):
        global multiplier
        numDisparities = cv2.getTrackbarPos('numDisparities','disp'+name)*16
        blockSize = cv2.getTrackbarPos('blockSize','disp'+name)*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp'+name)
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp'+name)*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp'+name)
        minDisparity = cv2.getTrackbarPos('minDisparity','disp'+name)
     

        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        return stereo, calculate_disparity(stereo, left_nice, right_nice, minDisparity, numDisparities)


def calculate_disparity(stereo, Left_nice, Right_nice, minDisparity, numDisparities):
            # Calculating disparity using the StereoBM algorithm
        rawdisparity = stereo.compute(Left_nice,Right_nice).astype(np.float32)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.
            
        # Converting to float32 
    
    
        # Scaling down the disparity values and normalizing them 
        disparity = (rawdisparity/16.0 - minDisparity)/numDisparities

        return disparity


def pointCloudFromDisparity(disparity, q):
    global count
    h, w = disparity.shape[:2]
    f=.8*w
    Q = np.float32([[1, 0, 0,      0],
                    [0,-1, 0,      0],
                    [0, 0, f*0.05, 100],
                    [0, 0, 0,      1]])
    

    pcl = o3d.geometry.PointCloud()

    
    mask = np.array((disparity > disparity.min()))
    

    point_cloud = cv2.reprojectImageTo3D(disparity, Q)

    point_cloud = point_cloud[mask]

    point_cloud= point_cloud.reshape(-1, 3)

  
    point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]

    
   
   
    #point_cloud = point_cloud[mask]

    #print(point_cloud, len(point_cloud), len(point_cloud[0]))

    pcl.points = o3d.utility.Vector3dVector(point_cloud)


    #print(np.asarray(pcl.points), len(pcl.points))

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
        f.write((ply_header % dict(vet_nrum=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def getGeo(rsm, M10, M02, M23, stereo, oldGeo=None, selection="M10"):
    frames = getRectifiedFrames(rsm, M10, M02, M23)
    unrectified = rsm.get_frames()
    if(selection == "M10"):
        left = frames[0]
        right = frames[1]
        color = unrectified[4]
    elif(selection == "M02"):
        left = frames[2]
        right = frames[3]
    else:
        left = frames[4]
        right = frames[5]
        color = unrectified[5]

    stereo, disp = updateOpenCV(stereo, left, right, selection)
    cv2.imshow("disp"+selection, disp)
    if(selection == "M10"):
        Q = M10[4]
    elif(selection == "M02"):
        Q = M02[4]
    else:
        Q = M23[4]

    global multiplier

    disp = disp * 6

    colored_points = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    if(oldGeo):
        oldGeo.points = pointCloudFromDisparity(disp, Q).points
    else:
        oldGeo = pointCloudFromDisparity(disp, Q)
    
    oldGeo.colors = o3d.utility.Vector3dVector(colored_points.reshape(-1, colored_points.shape[-1]))

    return stereo, oldGeo


def main():
    rsm = RealsenseManager()
    rsm.enableDevices()

    M10 = load_stereo_coefficients(1, 0)
    M02 = load_stereo_coefficients(0, 2)
    M23 = load_stereo_coefficients(2, 3)        # Displaying the disparity map


    stereo = cv2.StereoSGBM_create()
    setup_openCV("M10")
    setup_openCV("M23")

    #v = Visualizer()
    stereo, oldGeo1 = getGeo(rsm, M10, M02, M23, stereo, None, "M10")
    stereo, oldGeo2 = getGeo(rsm, M10, M02, M23, stereo, None, "M23")
    geometry = oldGeo1
    geometry.points.extend(oldGeo2.points)
    geometry.colors.extend(oldGeo2.colors)

    #v.do_once(geometry)

    

    extrinsic = create_extrinsic(Pose(-50, 0, 0, 0, 0.872665, 0))


    while True:

        print("~~~~~~~~~~~~~~~~~~~~~~~~~ LOOP ~~~~~~~~~~~~~~~~~~~~~~")
        
        stereo, Geo1 = getGeo(rsm, M10, M02, M23, stereo, oldGeo1, "M10")
        stereo, Geo2 = getGeo(rsm, M10, M02, M23, stereo, oldGeo2, "M23")
        geometry = oldGeo1
        geometry.points = Geo1.points
        geometry.colors = Geo1.colors
        geo2Points = []
        for point in tqdm(Geo2.points):
            homogenous_point = np.asarray([point[0], point[1], point[2], 1])
            new_point = np.dot(extrinsic, homogenous_point)
            #new_point = np.dot()
            unHomo = np.asarray([new_point[0], new_point[1], new_point[2]])
            #unHomo /= new_point[3]
            geo2Points.append(unHomo)
        nparr = np.asarray(geo2Points)
        a = o3d.geometry.PointCloud()
        a.points = o3d.utility.Vector3dVector(nparr)

        geometry.points.extend(a.points)
        geometry.colors.extend(Geo2.colors)

        #v.do_each_loop(geometry)
        
        oldGeo1 = Geo1
        oldGeo2 = Geo2
        
        global count
        if count == 0:
            np.save("pointcloudFromV1", np.asarray(geometry.points))
            print("saved")
            break
            
        count += 1 
        #Close window using esc key
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()