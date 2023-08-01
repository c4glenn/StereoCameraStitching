import cv2
import open3d as o3d
import open3d.visualization as vis
from enum import Enum
from realSenseWrapper import RealsenseManager
import numpy as np
from plySave import write_ply
from IntrinsicsAndExtrinsics import DEFUALT_EXTRINSICS, DEFAULT_INTRINSICS, CALIBRATED_EXTRINSICS, CALIBRATED_INTRINSICS

#SETABLE FLAGS
REALTIME = True
COLORS = False

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
    def __init__(self, method:RectificationMethod, rsm: RealsenseManager) -> None:
        self.method = method
        self.methodResolver = {
            RectificationMethod.CALIBRATED_INTRINSIC_AND_EXTRINSIC: (CALIBRATED_INTRINSICS, CALIBRATED_EXTRINSICS),
            RectificationMethod.CALIBRATED_EXTRINSIC: (DEFAULT_INTRINSICS, CALIBRATED_EXTRINSICS),
            RectificationMethod.CALIBRATED_INTRINSIC: (CALIBRATED_INTRINSICS, DEFUALT_EXTRINSICS),
            RectificationMethod.DEFAULT: (DEFAULT_INTRINSICS, DEFUALT_EXTRINSICS)
        }
        self.rsm = rsm
        
    
    def generatePointCloud(self, oldGeo=None):
        frames = self.rsm.get_frames()
        depth = self.rsm.getDepthFrames()
        geo = o3d.geometry.PointCloud() if not oldGeo else oldGeo

        frames = [frames[0], frames[3]]

        firstItterFlag = True

        intrin, extrin = self.methodResolver[self.method]

        for color, depth, intrinsic, extrinsic in zip(frames, depth, intrin, extrin):
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
        
        np.save(f"numpySaves/V2-{self.method}.ply", np.array(geo.points))
        
        return geo

    def generateOnePointCloud(self, color, depth, extrinsic, intrinsic):
        npdepth = np.asarray(depth)
        rgb = o3d.t.geometry.Image(color.astype(np.uint16))
        
        depth = o3d.t.geometry.Image(npdepth.astype(np.uint16))
        #is.draw_geometries([rgb.to_legacy()])
        rgbd = o3d.t.geometry.RGBDImage(rgb, depth, True)

        pointcloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)

        return pointcloud


def main():
    rsm = RealsenseManager(None, True, COLORS)
    rsm.enableDevices()
    recDef = Rectifier(RectificationMethod.DEFAULT, rsm)
    pointCloudDef = recDef.generatePointCloud()
    recCI = Rectifier(RectificationMethod.CALIBRATED_INTRINSIC, rsm)
    pointCloudCI = recCI.generatePointCloud()
    recCE = Rectifier(RectificationMethod.CALIBRATED_EXTRINSIC, rsm)
    pointCloudCE = recCE.generatePointCloud()
    recCIE = Rectifier(RectificationMethod.CALIBRATED_INTRINSIC_AND_EXTRINSIC, rsm)
    pointCloudCIE = recCIE.generatePointCloud()
    viz = Visualizer()
    viz.doOnce(pointCloudCIE)
    
    if REALTIME:
        while True:
            pointCloudDef = recDef.generatePointCloud(pointCloudDef)
            pointCloudCI = recCI.generatePointCloud(pointCloudCI)
            pointCloudCE = recCE.generatePointCloud(pointCloudCE)
            pointCloudCIE = recCIE.generatePointCloud(pointCloudCIE)
            write_ply("comparingPointClouds/Default.ply", np.asarray(pointCloudDef.points))
            write_ply("comparingPointClouds/CalibratedIntrinsic.ply", np.asarray(pointCloudCI.points))
            write_ply("comparingPointClouds/CalibratedExtrinsic.ply", np.asarray(pointCloudCE.points))
            write_ply("comparingPointClouds/CalibratedIntrinsicAndExtrinsic.ply", np.asarray(pointCloudCIE.points))
            viz.doEachLoop(pointCloudCIE)
        
    

if __name__ == "__main__":
    main()