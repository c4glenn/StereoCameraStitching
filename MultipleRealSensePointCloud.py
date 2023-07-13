import json
import open3d as o3d
import open3d.visualization as vis
import numpy as np
import datetime
from enum import Enum


DEFAULT_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
NUM_CAMERAS = 2


class Visualizer():
    def do_once(self, geometry):
        self.controller = vis.Visualizer()
        self.controller.create_window()
        self.controller.add_geometry(geometry)
        self.controller.poll_events()
        self.controller.update_renderer()

    def do_each_loop(self, geometry):
        self.controller.update_geometry(geometry)
        self.controller.poll_events()
        self.controller.update_renderer()


class RealSenseCamera:
    def __init__(self, config:o3d.t.io.RealSenseSensorConfig, intrinsic, extrinsic) -> None:
        self.config = config
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.capturing = False

        self.cam = o3d.t.io.RealSenseSensor()
        self.cam.init_sensor(self.config, filename=f"Saves/{self.config}-{datetime.datetime.now()}")
    
    def start_capture(self, record=False):
        self.cam.start_capture(record)
        self.capturing = True
    
    def stop(self):
        self.cam.stop_capture()

    def generate_point_cloud(self):
        if(not self.capturing):
            raise RuntimeError("Must start capture before generating pointclouds")
        capture = self.cam.capture_frame(True, True)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(capture.color.to_legacy(), capture.depth.to_legacy(), convert_rgb_to_intensity=False)
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsic, self.extrinsic)
    
    ############### STATIC METHODS ############

    def create_config_with_file(filename:str) -> o3d.t.io.RealSenseSensorConfig:
        with open(filename) as config_file:
            return o3d.t.io.RealSenseSenseorConfig(json.load(config_file))
        
    def create_config_with_serial(serial:str) -> o3d.t.io.RealSenseSensorConfig:
        with open("defaultRealSenseConfig.json") as config_file:
            config_settings = json.load(config_file)
            config_settings["serial"] = serial
            return o3d.t.io.RealSenseSensorConfig(config_settings)
    
    def create_config_with_index(index:int) -> o3d.t.io.RealSenseSensorConfig:
        devices = o3d.t.io.RealSenseSensor.enumerate_devices()
        return RealSenseCamera.create_config_with_serial(devices[index].serial)
    
class RectificationMethod(Enum):
    EXTRINSIC = 0

class Rectifier():
    def __init__(self, method: RectificationMethod) -> None:
        self.method = method
        self.rectification_methods = {
            RectificationMethod.EXTRINSIC: self.extrinsic_rectification 
        }

    
    def rectify(self, pointclouds: list[o3d.geometry.PointCloud], old_geo=None) -> o3d.geometry.PointCloud:
        return self.rectification_methods[self.method](pointclouds, old_geo)
    
    def extrinsic_rectification(self, pointclouds: list[o3d.geometry.PointCloud], old_geo = None) -> o3d.geometry.PointCloud:
        if not pointclouds:
            raise ValueError("Need to pass at least one pointcloud")
        
        if(len(pointclouds) == 1):
            return pointclouds[0]

        main_pointcloud = o3d.geometry.PointCloud() if not old_geo else old_geo
        main_pointcloud.points = pointclouds[0].points
        main_pointcloud.colors = pointclouds[0].colors

        for pointcloud in pointclouds[1:]:
            main_pointcloud.points.extend(pointcloud.points)
            main_pointcloud.colors.extend(pointcloud.colors)
        
        return main_pointcloud




if __name__ == "__main__":
    visualizer = Visualizer()
    rectifier = Rectifier(RectificationMethod.EXTRINSIC)
    cameras:list[RealSenseCamera] = []
    for i in range(NUM_CAMERAS):
        config = RealSenseCamera.create_config_with_index(i)
        cameras.append(RealSenseCamera(config, DEFAULT_INTRINSIC, np.identity(4)))
        cameras[i].start_capture(False)
    
    geo = rectifier.rectify([x.generate_point_cloud() for x in cameras])
    old_geo = geo
    visualizer.do_once(geo)
    try:
        while True:
            geo = rectifier.rectify([x.generate_point_cloud() for x in cameras], old_geo)
            old_geo = geo
            visualizer.do_each_loop(geo)
    except KeyboardInterrupt:
        for cam in cameras:
            cam.stop()
