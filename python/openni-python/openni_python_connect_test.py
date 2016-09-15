#!/usr/bin/env python

from primesense import openni2
import numpy as np
import matplotlib.pyplot as plt

def main():
    openni2.initialize('/usr/lib')

    dev = openni2.Device.open_any()
    print dev.get_sensor_info(openni2.SENSOR_DEPTH)

    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()

    depth_array = np.ndarray((frame.height,frame.width),dtype=np.uint16,buffer=frame_data)
    plt.imshow(depth_array)
    plt.show()

    depth_stream.stop()

    openni2.unload()



if __name__ == '__main__':
    main()
