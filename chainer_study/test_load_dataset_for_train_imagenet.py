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
import os
import glob
import re
import cv2

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean
        # image -= self.mean[:, top:bottom, left:right]
        image /= 255
        return image, label


def generate_dataset_list_file(root = os.path.join(os.getenv('HOME'), 'dataset/ILSVRC2012_img_train_t3/images/'),
                               output_filename = 'dataset_list.txt', sort = False):
    category_id_list = []
    output_file = open(os.path.join(root, output_filename), 'w')
    files = glob.glob(os.path.join(root, '*.JPEG'))
    if sort:
        files.sort()
    for f in files:
        img_shape = cv2.imread(f).shape
        print(f)
        if img_shape[0] <= 250 or img_shape[1] <= 250:
            continue
        # example of image filename: 'n02085620_7.JPEG'
        m = re.compile('n([0-9]+)\_([0-9]+)\.JPEG').search(os.path.basename(f))
        category_id = int(m.group(1))
        object_id = int(m.group(2))
        if not(category_id in category_id_list):
            category_id_list.append(category_id)
        category_id_from_0 = category_id_list.index(category_id)
        output_file.write(' '.join([f, str(category_id_from_0)]) + '\n')

def main():

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--train', type=str, default=os.path.join(
        os.getenv('HOME'), 'dataset/ILSVRC2012_img_train_t3/images/dataset_list.txt'),
                        help='Path to training image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default=os.path.join(
        os.getenv('HOME'), 'src/chainer/examples/imagenet/mean_for_alex.npy'),
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    args = parser.parse_args()

    show = True
    generate_list = False

    if generate_list:
        generate_dataset_list_file()

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean, 227)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    mean_img = np.zeros_like(train[0][0])
    for train_i in train:
        mean_img += train_i[0]
        print('category: {}'.format(train_i[1]))
        if show:
            cv2.imshow('test', cv2.cvtColor(train_i[0].transpose(1,2,0), cv2.COLOR_RGB2BGR))
            input = cv2.waitKey(0)
            if input == ord('q'):
                break
    mean_img = (mean_img * 255) / len(train)
    np.save('mean.npy', mean_img)
    if show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
