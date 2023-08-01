import pyrealsense2 as rs
import numpy as np
import cv2

USECAMERA = True
FILEPATH = "1689969928.504082.jpg"

class Device:
    def __init__(self, pipeline, pipeline_profile, product_line) -> None:
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.product_line = product_line

class RealsenseManager:
    def __init__(self, calibration=None, infrared:bool=False, color:bool=False) -> None:
        if(USECAMERA):
            self.calibration = calibration
            self.infrared = infrared
            self.color = color
            self.depthFrames = []
            
            self.context = rs.context()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
            self.config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 6)
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
            if color:
                self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
            
            self.enabled_devices: dict[int, Device] = {}
        else:
            pass

    def rectify(self, frame, serial, type, index):
        if(int(serial) == 927522071127):
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        return frame

    def getDepthFrames(self):
        if USECAMERA:
            return self.depthFrames
                

    def enableDevices(self):
        if USECAMERA:
            for device in self.context.devices:
                device.hardware_reset()
                pipeline = rs.pipeline()
                
                product = device.get_info(rs.camera_info.product_line)
                serial = device.get_info(rs.camera_info.serial_number)

                self.config.enable_device(serial)
                pipeline_profile = pipeline.start(self.config)

                print(self.context.sensors)

                sensor = pipeline_profile.get_device().first_depth_sensor()
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, 1 if self.infrared else 0)

                self.enabled_devices[serial] = Device(pipeline, pipeline_profile, product)
            
    def get_frames(self) -> list:
        if USECAMERA:
            return self.get_cam_frames()
        else:
            return [cv2.imread("SavedFrames/frame1/"+FILEPATH), cv2.imread("SavedFrames/frame2/"+FILEPATH), cv2.imread("SavedFrames/frame4/"+FILEPATH), cv2.imread("SavedFrames/frame5/"+FILEPATH), cv2.imread("SavedFrames/frame0/"+FILEPATH), cv2.imread("SavedFrames/frame3/"+FILEPATH)]
    
    
            
    def get_cam_frames(self) -> list:
        frames = {}
        setFrames = 0
        #print(f"{len(self.enabled_devices.items())} enabled devices")
        for serial, device in self.enabled_devices.items():
            streams = device.pipeline_profile.get_streams()
            frameset = device.pipeline.wait_for_frames()
            #print(f"streams: {len(streams)} framset: {frameset.size()}")
            if(frameset.size() == len(streams)):
                #print("framset was the number of streams ")
                frames[serial] = {}
                for stream in streams:
                    if(stream.stream_type() == rs.stream.infrared):
                        frame = frameset.get_infrared_frame(stream.stream_index())
                        key_ = (stream.stream_type(), stream.stream_index())
                    elif(stream.stream_type() == rs.stream.depth):
                        frame = frameset.first_or_default(stream.stream_type())
                        if(setFrames == 1):
                            self.depthFrames.append(self.rectify(np.asanyarray(frame.get_data()), serial, stream.stream_type(), stream.stream_index()))
                        else:
                            self.depthFrames = [self.rectify(np.asanyarray(frame.get_data()), serial, stream.stream_type(), stream.stream_index())]
                            setFrames = 1
                        key_ = (stream.stream_type(), 0)
                    else:
                        frame = frameset.first_or_default(stream.stream_type())
                        key_ = (stream.stream_type(), 0)
                    frames[serial][key_] = self.rectify(np.asanyarray(frame.get_data()), serial, stream.stream_type(), stream.stream_index())
        cleanFrames = []

        for serial, key in frames.items():
            for k, frame in key.items():
                if(k[0] == rs.stream.depth):
                    continue
                cleanFrames.append((f"{serial}{k[1]}", frame))
        cleanFrames.sort(key=lambda x: x[0])

        #print(f"{len(cleanFrames)} frames being outputed")

        return [f[1] for f in cleanFrames]

    
if __name__ == "__main__":
    rsm = RealsenseManager(None, True, True)
    rsm.enableDevices()
    while True:
        frame = rsm.get_frames()

        cv2.imshow("img", frame[0])
        k = cv2.waitKey(1)

        if(k == ord("q")): break


