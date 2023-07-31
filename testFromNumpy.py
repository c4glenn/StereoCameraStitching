import numpy as np
import open3d as o3d

val = np.load("pointcloudFromV2.npy")



pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(val)

o3d.visualization.draw_geometries([pcl])

