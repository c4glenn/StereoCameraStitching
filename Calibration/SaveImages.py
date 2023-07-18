import pyrealsense2 as rs
import cv2
import numpy as np
import os
import datetime

class Device:
    def __init__(self, pipeline, pipeline_profile, product_line) -> None:
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.product_line = product_line


def get_frames():
    frames = {}
    while len(frames) < len(enabled_devices.items()):
        for (serial, device) in enabled_devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.poll_for_frames()
            if(frameset.size() == len(streams)):
                dev_info = (serial, device.product_line)
                frames[dev_info] = {}

                for stream in streams:
                    if(rs.stream.infrared == stream.stream_type()):
                        frame = frameset.get_infrared_frame(stream.stream_index())
                        key_ = (stream.stream_type(), stream.stream_index())
                    else:
                        frame = frameset.first_or_default(stream.stream_type())
                        key_ = (stream.stream_type(), 0)
                    frames[dev_info][key_] = frame
    return frames

context = rs.context()

config = rs.config()
#config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 6)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 6)
#config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 6)


enabled_devices: dict[int, Device] = {}


for device in context.devices:
    pipeline = rs.pipeline()
    product = device.get_info(rs.camera_info.product_line)
    serial = device.get_info(rs.camera_info.serial_number)

    config.enable_device(serial)
    pipeline_profile = pipeline.start(config)

    sensor = pipeline_profile.get_device().first_depth_sensor()
    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, 0)

    enabled_devices[serial] = Device(pipeline, pipeline_profile, product)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER ,30, 0.001)

rows = 5
columns = 7
world_scaling = 2.54 # ish CM grid squares

objp = np.zeros((rows*columns, 3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1, 2)
objp = world_scaling * objp



while True:
    multi_cam_frames = get_frames()
    cols = []
    allImages = {}
    width = np.asanyarray(list(list(multi_cam_frames.values())[0].values())[0].get_data()).shape[1]
    height = np.asanyarray(list(list(multi_cam_frames.values())[0].values())[0].get_data()).shape[0]
    for cam, frames in multi_cam_frames.items():
        images = []
        for info, frame in frames.items():
            image = np.asanyarray(frame.get_data())
            if(info[0] == rs.stream.depth):
                image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_JET)
            else:
                image = cv2.resize(image, dsize=(480, 270), interpolation=cv2.INTER_AREA)
            if(int(cam[0]) == 927522071127):
                image = cv2.rotate(image, rotateCode=cv2.ROTATE_180)
            

            gray = None
            gray = cv2.copyTo(image, None, gray)

            ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

            if(ret==True):
                conv_size = (11, 11)

                corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
                cv2.drawChessboardCorners(gray, (rows, columns), corners, ret)

            images.append(gray)
            allImages[f"{cam[0]}-{'L' if info[1] == 1 else 'R'}"] = image
        cols.append(np.hstack(images))
    
    window = np.vstack(cols)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', window)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break
    if(key == ord('s')):
        time = datetime.datetime.now().timestamp()
        for info, image in allImages.items():
            try:
                os.mkdir(f"Calibration/images/{info}")
            except FileExistsError:
                pass

            cv2.imwrite(f"Calibration/images/{info}/{time}.jpg", image)
