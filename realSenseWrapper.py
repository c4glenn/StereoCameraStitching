import pyrealsense2 as rs
import numpy as np


class Device:
    def __init__(self, pipeline, pipeline_profile, product_line) -> None:
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.product_line = product_line

class RealsenseManager:
    def __init__(self, calibration, infrared:bool=False, color:bool=False) -> None:
        self.calibration = calibration
        self.infrared = infrared
        self.color = color
        
        self.context = rs.context()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
        self.config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 6)
        if color:
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
        
        self.enabled_devices: dict[int, Device] = {}

    def rectify(self, frame, serial, type, index):
        return frame


    def enableDevices(self):
        for device in self.context.devices:
            pipeline = rs.pipeline()
            product = device.get_info(rs.camera_info.product_line)
            serial = device.get_info(rs.camera_info.serial_number)

            self.config.enable_device(serial)
            pipeline_profile = pipeline.start(self.config)

            sensor = pipeline_profile.get_device().first_depth_sensor()
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1 if self.infrared else 0)

            self.enabled_devices[serial] = Device(pipeline, pipeline_profile, product)
        
    def get_frames(self) -> list:
        frames = {}
        while len(frames) < len(self.enabled_devices.items()):
            for serial, device in self.enabled_devices.items():
                streams = device.pipeline_profile.get_streams()
                frameset = device.pipeline.poll_for_frames()
                if(frameset.size() == len(streams)):
                    frames[serial] = {}
                    for stream in streams:
                        if(stream.stream_type() == rs.stream.infrared):
                            frame = frameset.get_infrared_frame(stream.stream_index())
                            key_ = (stream.stream_type(), stream.stream_index())
                        else:
                            frame = frameset.first_or_default(stream.stream_type())
                            key_ = (stream.stream_type(), 0)
                        frames[serial][key_] = self.rectify(frame, serial, stream.stream_type(), stream.stream_index())
        cleanFrames = []

        for serial, key in frames.items():
            for k, frame in key.items():
                cleanFrames.append(np.asanyarray(frame.get_data()))

        return cleanFrames

    
if __name__ == "__main__":
    rsm = RealsenseManager(None, False, True)
    rsm.enableDevices()
    frame = rsm.get_frames()

    print(frame)


