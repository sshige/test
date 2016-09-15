#!/usr/bin/env python

import numpy as np
import cv2
import sys
import os
import math

def find_file_dir(search_dir, filename):
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f == filename:
                return root

def draw_rectangle(rect_filepath, img, color1, color2):
    f = open(rect_filepath, 'r')
    poslist = list(f)
    for i in range(len(poslist) / 4):
        rect_vertices = [[float(poslist[i*4 + k].split(' ')[j]) for j in range(2)] for k in range(4)]
        if not any([math.isnan(item) for sublist in rect_vertices for item in sublist]):
            for j in range(4):
                rect_v1 = tuple(int(rect_vertices[j % 4][k]) for k in range(2))
                rect_v2 = tuple(int(rect_vertices[(j+1) % 4][k]) for k in range(2))
                if j%2 == 0:
                    rect_color = color1
                else:
                    rect_color = color2
                cv2.line(img, rect_v1, rect_v2, rect_color, 2)
    return img

if __name__ == '__main__':
    param = sys.argv

    if len(param) > 1:
        dataset_dir = param[1]
    else:
        dataset_dir = '/home/leus/Desktop/CornellGraspingDataset/raw'

    for idx in range(0000, 1035):
        # set filename
        idx = "{0:04d}".format(idx)
        image_filename = 'pcd' + idx + 'r.png'
        prect_filename = 'pcd' + idx + 'cpos.txt'
        nrect_filename = 'pcd' + idx + 'cneg.txt'

        # search filename
        dirpath = find_file_dir(dataset_dir, image_filename)

        if dirpath:
            # get full path of files
            image_filepath = os.path.join(dirpath, image_filename)
            prect_filepath = os.path.join(dirpath, prect_filename)
            nrect_filepath = os.path.join(dirpath, nrect_filename)

            # load image
            print("load file: {}".format(image_filepath))
            img = cv2.imread(image_filepath)
            img_with_rect = img.copy()

            # draw rectangle
            img_with_rect = draw_rectangle(prect_filepath, img_with_rect, color1 = (0, 255, 0), color2 = (0, 0, 255))
            img_with_rect = draw_rectangle(nrect_filepath, img_with_rect, color1 = (0, 255, 255), color2 = (255, 0, 255))

            # show image with rectangle
            cv2.imshow('image', img_with_rect)
            quit_flag = False
            draw_rect_flag = True
            while True:
                input = cv2.waitKey(0)
                if input == ord('d'):
                    draw_rect_flag = not(draw_rect_flag)
                    if draw_rect_flag:
                        cv2.imshow('image', img_with_rect)
                    else:
                        cv2.imshow('image', img)
                elif input == ord('q'):
                    quit_flag = True
                    break
                else:
                    break
            if quit_flag:
                break

    cv2.destroyAllWindows()
