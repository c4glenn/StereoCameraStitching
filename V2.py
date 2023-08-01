import cv2
import open3d as o3d
import open3d.visualization as vis
from enum import Enum
from realSenseWrapper import RealsenseManager
from dataclasses import dataclass
import numpy as np
from plySave import write_ply

from numpy import cos, sin, pi


#SETABLE FLAGS
REALTIME = True
COLORS = False
TRANSLATIONAL_SCALE:float = 0.05


CALIBRATED_INTRINSICS = [
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

CALIBRATED_EXTRINSICS = [
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
    DEFAULT = 0
    CALIBRATED_INTRINSIC = 1
    CALIBRATED_EXTRINSIC = 2
    CALIBRATED_INTRINSIC_AND_EXTRINSIC = 3

class Rectifier:
    def __init__(self, method:RectificationMethod) -> None:
        self.method = method
        self.methodResolver = {
            RectificationMethod.CALIBRATED_INTRINSIC_AND_EXTRINSIC: self.calibratedIntrinsicAndExtrinsic,
            RectificationMethod.CALIBRATED_EXTRINSIC: self.calibratedExtrinsic,
            RectificationMethod.CALIBRATED_INTRINSIC: self.calibratedIntrinsic,
            RectificationMethod.DEFAULT: self.default
        }
        self.rsm = RealsenseManager(None, True, COLORS)
        self.rsm.enableDevices()
        
    
    def generatePointCloud(self, oldGeo=None):
        frames = self.rsm.get_frames()
        depth = self.rsm.getDepthFrames()
        geo = o3d.geometry.PointCloud() if not oldGeo else oldGeo

        frames = [frames[0], frames[3]]

        return self.methodResolver[self.method](geo, frames, depth)

    def generateOnePointCloud(self, color, depth, extrinsic, intrinsic):
        npdepth = np.asarray(depth)
        rgb = o3d.t.geometry.Image(color.astype(np.uint16))
        
        depth = o3d.t.geometry.Image(npdepth.astype(np.uint16))
        #is.draw_geometries([rgb.to_legacy()])
        rgbd = o3d.t.geometry.RGBDImage(rgb, depth, True)

        pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return pointcloud
    
    def calibratedExtrinsic(self, geo, frames, depth):
        pass
    
    def calibratedIntrinsic(self, geo, frames, depth):
        pass
    
    def default(self, geo, frames, depth):
        pass
    
    def calibratedIntrinsicAndExtrinsic(self, geo, frames, depth):
        firstItterFlag = True

        for color, depth, intrinsic, extrinsic in zip(frames, depth, CALIBRATED_INTRINSICS, CALIBRATED_EXTRINSICS):
            pc = self.generateOnePointCloud(color, depth, extrinsic, intrinsic)
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
        
        return geo

def main():
    rec = Rectifier(RectificationMethod.CALIBRATED_INTRINSIC_AND_EXTRINSIC)
    viz = Visualizer()
    pointCloud = rec.generatePointCloud()
    viz.doOnce(pointCloud)
    
    if REALTIME:
        while True:
            pointCloud = rec.generatePointCloud(pointCloud)
            write_ply("comparingPointClouds/CalibratedIntrinsicAndExtrinsic", np.asarray(pointCloud.points))
            viz.doEachLoop(pointCloud)
        
    

if __name__ == "__main__":
    main()