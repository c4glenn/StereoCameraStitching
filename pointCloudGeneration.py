import cv2
import open3d as o3d
import open3d.visualization as vis
from enum import Enum
from realSenseWrapper import RealsenseManager
from dataclasses import dataclass
import numpy as np

from numpy import cos, sin, pi


#SETABLE FLAGS
REALTIME = True
COLORS = True
TRANSLATIONAL_SCALE = 25.4


CAMERA_INTRINSICS = [
    np.array([
        [642.69646676,   0,         646.336484],
        [0,         644.92030593, 351.69780459],
        [0, 0, 1]
    ]),
    np.array([
        [639.53915538,   0,         642.22773266],
        [0,         641.38228746, 364.04678063],
        [0, 0, 1]
    ])
]

CAMERA_EXTRINSICS = [
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),
    np.array([
        [ 9.99999991e-01,  3.22346979e-06, 1.31991937e-04, -1.71120036],
        [-3.08123524e-06,  9.99999419e-01, -1.07758629e-03, 0.00405337],
        [-1.31995334e-04,  1.07758588e-03,  9.99999411e-01, -0.06324388],
        [0, 0, 0, 1]
    ])
]


@dataclass
class Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    rX: float = 0
    rY: float = 0
    rZ: float = 0

def load_stereo_coefficients(number1, number2):
    cv_file = cv2.FileStorage("improved_params2.xml", cv2.FileStorage_READ)
    leftMapX = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_x").mat()
    leftMapY = cv_file.getNode(f"M{number1}-{number2}L_Stereo_Map_y").mat()
    rightMapX = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_x").mat()
    rightMapY = cv_file.getNode(f"M{number1}-{number2}R_Stereo_Map_y").mat()
    Q = cv_file.getNode(f"M{number1}-{number2}Q").mat()
    cv_file.release()
    return leftMapX, leftMapY, rightMapX, rightMapY, Q


class Visualizer:
    def __init__(self) -> None:
        self.controller = vis.Visualizer()
        self.controller.create_window()
        self.geometry = o3d.geometry.PointCloud()

        self.controller.add_geometry(self.geometry)
        self.controller.poll_events()
        self.controller.update_renderer()
    
    def displayImage(self, name, image):
        cv2.imshow(name, image)
    
    def doEachLoop(self, pointcloud):        
        print(f"passed in:{len(pointcloud.points)} points")
        
        self.geometry.points = pointcloud.points

        if COLORS:
            self.geometry.colors = pointcloud.colors

        np.save("pointcloudFromV2.npy", np.asarray(self.geometry.points))
        print("saved")
        
        self.controller.update_geometry(self.geometry)
        self.controller.poll_events()
        self.controller.update_renderer()

class RectificationMethod(Enum):
    V0 = 0
    V1 = 1


CAMERA_POSES = [
    Pose(x=0, y=0, z=0, rX=0, rY=0, rZ=0),
    Pose(x=0, y=0, z=0, rX=0, rY=-0.872665, rZ=pi)
]

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
    
    
class Rectifier:
    def __init__(self, method:RectificationMethod) -> None:
        self.method = method
        self.methodResolver = {
            RectificationMethod.V0: self.V0generate,
            RectificationMethod.V1: self.V1generate
        }
        self.rsm = RealsenseManager(None, True, COLORS)
        self.rsm.enableDevices()
        
    
    def generatePointCloud(self):
        return self.methodResolver[self.method]()

    def generateOnePointCloudV0(self, color, depth, extrinsic, intrinsic):
        npdepth = np.asarray(depth)
        rgb = o3d.t.geometry.Image(color.astype(np.uint16))
        depth = o3d.t.geometry.Image(npdepth.astype(np.uint16))
        
        rgbd = o3d.t.geometry.RGBDImage(rgb, depth, True)

        pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return pointcloud

    
    def V0generate(self):
        print("V0 Selected to generate Point Clouds")
        frames = self.rsm.get_frames()
        print("Frames gotten")
        depth = self.rsm.getDepthFrames()
        print("depth gotten")
    

        geo = o3d.geometry.PointCloud() 

        for color, depth, intrinsic, extrinsic in zip(frames, depth, CAMERA_INTRINSICS, CAMERA_EXTRINSICS):
            pc = self.generateOnePointCloudV0(color, depth, extrinsic, intrinsic)
            legacyPC = pc.to_legacy()
            geo.points.extend(legacyPC.points)
            if COLORS:
                geo.colors.extend(legacyPC.colors)
            
                
        return geo
        
    
    def V1generate(self):
        pass

def main():
    rec = Rectifier(RectificationMethod.V0)
    viz = Visualizer()
    print("visualizer started")
    pointCloud = rec.generatePointCloud()
    print("point cloud generated")
    viz.doEachLoop(pointCloud)
    print("displayed")
    
    if REALTIME:
        while True:
            print("~~~~~~~~~~~~~~~~~LOOP~~~~~~~~~~~~~~~~~~")
            pointCloud = rec.generatePointCloud()
            print("point cloud generated")
            viz.doEachLoop(pointCloud)
            print("displayed")
        
    

if __name__ == "__main__":
    main()