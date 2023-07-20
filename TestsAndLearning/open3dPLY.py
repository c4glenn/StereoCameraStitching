import open3d as o3d


geo = o3d.io.read_point_cloud("TestsAndLearning/out.ply", remove_nan_points=True, remove_infinite_points=True)

o3d.visualization.draw_geometries([geo])