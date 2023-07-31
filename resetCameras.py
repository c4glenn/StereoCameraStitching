import pyrealsense2 as rs


def main():
    context = rs.context()
    for device in context.devices:
        device.hardware_reset()



if __name__ == "__main__":
    main()