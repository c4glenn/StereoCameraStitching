import json
import open3d as o3d
import open3d.visualization as vis
import cv2
import numpy as np

from time import sleep


o3d.t.io.RealSenseSensor.list_devices()


with open("realsenseConfig.json") as cf:
    a = json.load(cf)
    rs1_cfg = o3d.t.io.RealSenseSensorConfig(a)
    
with open("realsenseConfig copy.json") as cf:
    a = json.load(cf)
    rs2_cfg = o3d.t.io.RealSenseSensorConfig(a)


rs1 = o3d.t.io.RealSenseSensor()
rs1.init_sensor(rs1_cfg)
rs1.start_capture(False)

rs2 = o3d.t.io.RealSenseSensor()
rs2.init_sensor(rs2_cfg)
rs2.start_capture(False)

camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

visual = vis.Visualizer()
visual.create_window()

cam1_extrinsic = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

cam2_extrinsic = np.array([
    [np.cos(np.pi), -np.sin(np.pi), 0, -.5],
    [np.sin(np.pi), np.cos(np.pi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def createPointCloud(rgbdIM, extrinsic):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgbdIM.color.to_legacy(), depth=rgbdIM.depth.to_legacy(), convert_rgb_to_intensity=False)
    return o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=camera_intrinsic, extrinsic=extrinsic)


im_rgbd1 = rs1.capture_frame(True, True)
im_rgbd2 = rs2.capture_frame(True, True)
pointcloudCam1 = createPointCloud(im_rgbd1, cam1_extrinsic)
pointcloudCam2 = createPointCloud(im_rgbd2, cam2_extrinsic)
geometry = pointcloudCam1

geometry.points.extend(pointcloudCam2.points)
geometry.colors.extend(pointcloudCam2.colors)

visual.add_geometry(geometry)
visual.poll_events()
visual.update_renderer()

while True:
    #visual.remove_geometry(geometry, False)
    im = rs1.capture_frame(True, True)
    im2 = rs2.capture_frame(True, True)
    pointcloud = createPointCloud(im, cam1_extrinsic)
    pointcloud2 = createPointCloud(im2, cam2_extrinsic)
    
    geometry.points = pointcloud.points
    geometry.colors = pointcloud.colors
    geometry.points.extend(pointcloud2.points)
    geometry.colors.extend(pointcloud2.colors)


    visual.update_geometry(geometry)
    visual.poll_events()
    visual.update_renderer()

    #vis.draw_geometries([geometry])

rs1.stop_capture()

