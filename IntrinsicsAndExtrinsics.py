from dataclasses import dataclass
from numpy import pi, cos, sin
import numpy as np
import open3d as o3d

@dataclass
class Pose:
    x: float = 0
    y: float = 0
    z: float = 0
    rX: float = 0
    rY: float = 0
    rZ: float = 0

camera_poses = [
    Pose(x=0, y=0, z=0, rX=0, rY=0, rZ=0),
    Pose(x=0, y=0, z=0, rX=0, rY=-0.872665, rZ=pi)
]


def create_extrinsic(pose: Pose) -> np.array:
        ca, a, sa = cos(pose.rX), pose.rX, sin(pose.rX)
        cb, b, sb = cos(pose.rY), pose.rY, sin(pose.rY)
        cc, c, sc = cos(pose.rZ), pose.rZ, sin(pose.rZ)
        x = pose.x * 25.4
        y = pose.y * 25.4
        z = pose.z * 25.4
        return np.array([
            [cb*cc, -cb*sc, sb, x*cb*cc - y*cb*sc],
            [sa*sb*cc + ca*sc, ca*cc-sa*sb*sc, -sa*cb, x*(sa*sb*cc+ca*sc) + y*(ca*cc-sa*sb*sc) - z*(sa*cb)],
            [sa*sc-ca*sb*cc, ca*sb*sc+sa*cc, ca*cb, x*(sa*sc-ca*sb*cc)+y*(ca*sb*sc+sa*cc) + z*(ca*cb)],
            [0, 0, 0, 1]
        ])



DEFUALT_EXTRINSICS = [create_extrinsic(pose) for pose in camera_poses]
DEFAULT_INTRINSICS = [o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)] * 6


CALIBRATED_INTRINSICS = [
    np.array([
        [642.69646676,   0,         646.336484],
        [0,         644.92030593, 351.69780459],
        [0, 0, 1]
    ]),
    np.array([
        [639.53915538,   0,         642.22773266],
        [0,         641.38228746, 364.04678063],
        [0, 0, 1]
    ])
]

TRANSLATIONAL_SCALE = 0.05
CALIBRATED_EXTRINSICS = [
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]),
    np.array([
        [0.74754093,  0.01511931, -0.66404365, -2.14950322*TRANSLATIONAL_SCALE],
        [-0.00917135,  0.99988054,  0.01244127,0.02752463*TRANSLATIONAL_SCALE],
        [0.66415243, -0.00321018,  0.74759029,-0.79388597*TRANSLATIONAL_SCALE],
        [0, 0, 0, 1]
    ])
]