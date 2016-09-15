#!/usr/bin/env python

from primesense import openni2
import numpy as np
import cv2


if __name__ == '__main__':
    # initialize device
    openni2.initialize('/usr/lib')

    dev = openni2.Device.open_any()
    print dev.get_sensor_info(openni2.SENSOR_DEPTH)

    # initialize stream
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    depth_stream.start()
    color_stream.start()

    shot_idx = 0
    depth_array_draw_scale = 10000.0

    # loop
    while True:
        # get depth
        depth_frame = depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        depth_array = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16, buffer=depth_frame_data)

        # get rgb
        color_frame = color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_uint8()
        rgb_array = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8, buffer=color_frame_data)
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HSV)

        # process
        center_y = depth_array.shape[0]/2
        center_x = depth_array.shape[1]/2
        ## depth
        center_detph = depth_array[center_y][center_x]
        center_detph_thre = 1000
        if center_detph < center_detph_thre:
            # print "object exists in center. depth: %4d." % center_detph
            cv2.circle(depth_array, (center_x, center_y), 10, depth_array_draw_scale, 3)
        else:
            cv2.circle(depth_array, (center_x, center_y), 10, 0, 3)
        ## rgb
        center_rgb = rgb_array[center_y][center_x]
        center_hsv = hsv_array[center_y][center_x]
        center_hue = center_hsv[0]
        center_saturation = center_hsv[1]
        hue_red = 0
        hue_green = 60
        hue_blue = 120
        hue_thre = 20
        saturation_thre = 100
        print "center rgb: %s  hsv: : %s" % (center_rgb, center_hsv)
        if center_saturation > saturation_thre:
            if abs(center_hue - hue_green) < hue_thre:
                cv2.circle(rgb_array, (center_x, center_y), 10, (0, 255, 0), 3)
            elif abs(center_hue - hue_blue) < hue_thre:
                cv2.circle(rgb_array, (center_x, center_y), 10, (255, 0, 0), 3)
            elif abs(center_hue - hue_red) < hue_thre or abs(center_hue - (hue_red+180)) < hue_thre:
                cv2.circle(rgb_array, (center_x, center_y), 10, (0, 0, 255), 3)
            else:
                cv2.circle(rgb_array, (center_x, center_y), 10, (255, 255, 255), 3)
        else:
            cv2.circle(rgb_array, (center_x, center_y), 10, (0, 0, 0), 3)

        # show
        cv2.imshow('Depth', depth_array / depth_array_draw_scale) # 0-10000mm to 0.0-1.0
        cv2.imshow('Color', rgb_array)
        ch = 0xFF & cv2.waitKey(10)

        # keyboard
        if ch == 27:
            break
        elif ch == ord(' '):
            filename =  'shot_depth_%03d.png' % shot_idx
            cv2.imwrite(filename, (depth_array*255).astype(np.uint8))
            print filename, 'saved.'
            filename =  'shot_color_%03d.png' % shot_idx
            cv2.imwrite(filename, (rgb_array).astype(np.uint8))
            print filename, 'saved.'
            shot_idx += 1

    # finalize
    depth_stream.stop()
    openni2.unload()
