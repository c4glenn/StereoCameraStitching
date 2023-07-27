import cv2
import open3d as o3d
import open3d.visualization as vis
from enum import Enum
from realSenseWrapper import RealsenseManager

#SETABLE FLAGS
REALTIME = False
COLORS = False

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
        self.geometry.points = pointcloud.points
        if(COLORS):
            self.geometry.colors = pointcloud.colors
        
        self.controller.update_geometry(self.geometry)
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
        
        def generatePointCloud(self):
        self.methodResolver[self.method]()
    
    def V0generate(self):
        frames = self.rsm.get_frames()
        depth = self.rsm.getDepthFrames()
        
        
    
    def V1generate():
        pass

def main():
    rec = Rectifier(RectificationMethod.V1)
    viz = Visualizer()
    pointCloud = rec.generatePointCloud()
    viz.doEachLoop(pointCloud)
    
    if REALTIME:
        while True:
            rec.generatePointCloud()
        
        
    

if __name__ == "__main__":
    main()