import json
import open3d as o3d
import open3d.visualization as vis
import numpy as np
from numpy import sin, cos, pi
import datetime
from enum import Enum
from dataclasses import dataclass
import cv2

@dataclass
class Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    rX: float = 0
    rY: float = 0
    rZ: float = 0

DEFAULT_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
NUM_CAMERAS = 2
TRANSLATIONAL_SCALE = 1

CAMERA_POSES = [
    Pose(x=0, y=0, z=0, rX=0, rY=0, rZ=0),
    Pose(x=0, y=0, z=0, rX=0, rY=-0.872665, rZ=pi)
]

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
        self.count = 0

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
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsic, self.extrinsic)
        a = o3d.geometry.PointCloud()
        points = np.asarray(point_cloud.points)

        if self.count == 10:
            np.save("pointcloudFromV0", points)

        self.count += 1

        a.points = o3d.utility.Vector3dVector(points)
        a.colors = point_cloud.colors
        return a

    ############### STATIC METHODS ############

    def create_config_with_file(filename:str) -> o3d.t.io.RealSenseSensorConfig:
        with open(filename) as config_file:
            return o3d.t.io.RealSenseSensorConfig(json.load(config_file))

    def create_config_with_serial(serial:str) -> o3d.t.io.RealSenseSensorConfig:
        with open("defaultRealSenseConfig.json") as config_file:
            config_settings = json.load(config_file)
            config_settings["serial"] = serial
            return o3d.t.io.RealSenseSensorConfig(config_settings)
    
    def create_config_with_index(index:int) -> o3d.t.io.RealSenseSensorConfig:
        devices = o3d.t.io.RealSenseSensor.enumerate_devices()
        return RealSenseCamera.create_config_with_serial(devices[index].serial)
    
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
        pass
    
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


def create_cameras() -> list[RealSenseCamera]:
    cameras:list[RealSenseCamera] = []
    for i in range(NUM_CAMERAS):
        config = RealSenseCamera.create_config_with_index(i)
        extrinsic = RealSenseCamera.create_extrinsic(CAMERA_POSES[i])
        cameras.append(RealSenseCamera(config, DEFAULT_INTRINSIC, extrinsic))
        cameras[i].start_capture(False)
    
    return cameras

def main():
    visualizer = Visualizer()
    rectifier = Rectifier(RectificationMethod.EXTRINSIC)
    cameras = create_cameras()
    
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



if __name__ == "__main__":
    main()