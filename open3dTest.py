import json
import open3d as o3d


#o3d.t.io.RealSenseSensor.list_devices()


with open("realsenseConfig.json") as cf:
    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))


rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0)
rs.start_capture(True)
for fid in range(150):
    im_rgbd = rs.capture_frame(True, True)
    

rs.stop_capture()

