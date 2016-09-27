#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random
import six
import os
import numpy as np
import cv2

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import dataset_mixin
import chainer.functions as F
import chainer.links as L

from train_eus_cube_pose_util import *

def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('val', help='Path to training image-label list file')
    parser.add_argument('initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Image mean file')
    args = parser.parse_args()

    # Initialize the model to train
    model = ModelForImage2CubePose()
    chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    mean = None
    if args.mean:
        mean = np.load(args.mean)

    val = ImageRpyDataset(args.val, mean=mean)

    for val_i in val:
        input_image = chainer.cuda.to_gpu(val_i[0].reshape(1,3,120,120))
        ret = model(chainer.Variable(input_image)).data[0]
        truth = val_i[1]
        raw_image = val_i[0]
        if mean is not None:
            raw_image += mean
        raw_image = raw_image.transpose((1,2,0))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        raw_image = cv2.resize(raw_image, (400, 400))
        cv2.imshow('input',raw_image)
        print('ret: {}, truth: {}'.format(ret, truth))
        print('(test (list {} {} {}) (list {} {} {}))'.format(
            truth[0], truth[1], truth[2], ret[0], ret[1], ret[2]))
        input = cv2.waitKey(0)
        if input == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
