_, self.camera_matrix, self.distortion, _, _ = \
            cv2.calibrateCamera(self.object_points, self.image_points, self.image_size, None, None)

error, _, _, _, _, self.rotation, self.translation, _, _ = \
            cv2.stereoCalibrate(self.camera_left.object_points,
                                self.camera_left.image_points,
                                self.camera_right.image_points,
                                self.camera_left.camera_matrix,
                                self.camera_left.distortion,
                                self.camera_right.camera_matrix,
                                self.camera_right.distortion,
                                self.camera_left.image_size,
                                flags=cv2.CALIB_FIX_INTRINSIC)

self.rotation_left, self.rotation_right, self.perspective_left, self.perspective_right, self.Q, self.roi_left, self.roi_right = \
            cv2.stereoRectify(self.camera_left.camera_matrix,
                              self.camera_left.distortion,
                              self.camera_right.camera_matrix,
                              self.camera_right.distortion,
                              self.camera_left.image_size,
                              self.rotation,
                              self.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY)

self.rotation_left, self.rotation_right, self.perspective_left, self.perspective_right, self.Q, self.roi_left, self.roi_right = \
            cv2.stereoRectify(self.camera_left.camera_matrix,
                              self.camera_left.distortion,
                              self.camera_right.camera_matrix,
                              self.camera_right.distortion,
                              self.camera_left.image_size,
                              self.rotation,
                              self.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY)

self.block_matching = cv2.StereoSGBM().create() # params not mentioned here, but they are set
disparity = self.block_matching.compute(undistorted_left, undistorted_right)
disparity = cv2.convertScaleAbs(disparity, beta=16)

point_cloud = cv2.reprojectImageTo3D(disparity_image, Q)
point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])
point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]

pcl = open3d.geometry.PointCloud()
pcl.points = open3d.utility.Vector3dVector(point_cloud)
open3d.visualization.draw_geometries([pcl])
