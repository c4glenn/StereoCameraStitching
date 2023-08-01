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
TRANSLATIONAL_SCALE:float = 0.05


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
        [0.74754093,  0.01511931, -0.66404365, -2.14950322*TRANSLATIONAL_SCALE],
        [-0.00917135,  0.99988054,  0.01244127,0.02752463*TRANSLATIONAL_SCALE],
        [0.66415243, -0.00321018,  0.74759029,-0.79388597*TRANSLATIONAL_SCALE],
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
    
    def doOnce(self, geometry) -> None:
        self.controller.add_geometry(geometry)
        self.controller.poll_events()
        self.controller.update_renderer()

    def displayImage(self, name, image):
        cv2.imshow(name, image)
    
    def doEachLoop(self, pointcloud):                
        self.controller.update_geometry(pointcloud)
        self.controller.poll_events()
        self.controller.update_renderer()

class RectificationMethod(Enum):
    V0 = 0
    V1 = 1

class Rectifier:
    def __init__(self, method:RectificationMethod) -> None:
        self.method = method
        self.methodResolver = {
            RectificationMethod.V0: self.V0generate,
            RectificationMethod.V1: self.V1generate
        }
        self.rsm = RealsenseManager(None, True, COLORS)
        self.rsm.enableDevices()
        
    
    def generatePointCloud(self, oldGeo=None):
        return self.methodResolver[self.method](oldGeo)

    def generateOnePointCloudV0(self, color, depth, extrinsic, intrinsic):
        npdepth = np.asarray(depth)
        rgb = o3d.t.geometry.Image(color.astype(np.uint16))
        
        depth = o3d.t.geometry.Image(npdepth.astype(np.uint16))
        #is.draw_geometries([rgb.to_legacy()])
        rgbd = o3d.t.geometry.RGBDImage(rgb, depth, True)

        pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return pointcloud

    
    def V0generate(self, oldGeo):
        #print("V0 Selected to generate Point Clouds")
        frames = self.rsm.get_frames()
        
        #print(f"{len(frames)} Frames gotten")
        depth = self.rsm.getDepthFrames()

        #print(f"{len(depth)} depth frames gotten")
    

        geo = o3d.geometry.PointCloud() if not oldGeo else oldGeo

        frames = [frames[0], frames[3]]

        firstItterFlag = True

        for color, depth, intrinsic, extrinsic in zip(frames, depth, CAMERA_INTRINSICS, CAMERA_EXTRINSICS):
            pc = self.generateOnePointCloudV0(color, depth, extrinsic, intrinsic)
            legacyPC = pc.to_legacy()
            if(firstItterFlag):
                geo.points = legacyPC.points
                if(COLORS):
                    geo.colors = legacyPC.colors 
                firstItterFlag = False
            else:
                geo.points.extend(legacyPC.points)
                if COLORS:
                    geo.colors.extend(legacyPC.colors)
        

        np.save("pointcloudFromV2.npy", np.asarray(geo.points))
            
                
        return geo
        
    
    def V1generate(self):
        pass

def main():
    rec = Rectifier(RectificationMethod.V0)
    viz = Visualizer()
    pointCloud = rec.generatePointCloud()
    viz.doOnce(pointCloud)
    
    if REALTIME:
        while True:
            pointCloud = rec.generatePointCloud(pointCloud)
            viz.doEachLoop(pointCloud)
        
    

if __name__ == "__main__":
    main()