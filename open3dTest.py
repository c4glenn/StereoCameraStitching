import json
import open3d as o3d
import open3d.visualization as vis
import cv2
import numpy as np


#o3d.t.io.RealSenseSensor.list_devices()


with open("realsenseConfig.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))


rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
rs.start_capture(False)
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

for fid in range(150):
    im_rgbd = rs.capture_frame(True, True)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=o3d.geometry.RGBDImage.create_from_color_and_depth(color=im_rgbd.color.to_legacy(), depth=im_rgbd.depth.to_legacy(), convert_rgb_to_intensity=False), intrinsic=camera_intrinsic, extrinsic=np.identity(4))
    vis.draw_geometries([point_cloud])

rs.stop_capture()

