import numpy as np
import open3d as o3d

val = np.load("pointcloudFromV2.npy")



pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(val)

pcl.remove_duplicated_points()
pcl.remove_non_finite_points(True, True)
pcl.remove_statistical_outlier(10, .5, True)

o3d.visualization.draw_geometries([pcl])

