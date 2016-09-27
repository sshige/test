from __future__ import print_function
import argparse
import random
import six
import os
import numpy as np

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import dataset_mixin
import chainer.functions as F
import chainer.links as L


class ModelForImage2CubePose(chainer.Chain):

    wscale = 0.1

    def __init__(self):
        super(ModelForImage2CubePose, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=2, wscale=self.wscale),
            conv2=L.Convolution2D(96, 256,  5, pad=1, wscale=self.wscale),
            conv3=L.Convolution2D(256, 384,  3, pad=1, wscale=self.wscale),
            conv4=L.Convolution2D(384, 384,  3, pad=1, wscale=self.wscale),
            conv5=L.Convolution2D(384, 256,  3, pad=1, wscale=self.wscale),
            fc6=L.Linear(9216, 4096, wscale=self.wscale),
            fc7=L.Linear(4096, 4096, wscale=self.wscale),
            fc8=L.Linear(4096, 3, wscale=self.wscale),
        )
        self.train = True

    def __call__(self, x, t=None):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = F.tanh(self.fc8(h)) * np.pi

        self.score = h
        if t is None:
            return h
        else:
            loss = F.mean_squared_error(h, t)
            chainer.report({'loss': loss, 'score': self.score.data}, self)
            return loss


class ImageRpyDataset(dataset_mixin.DatasetMixin):

    def __init__(self, pairs, root='.', mean=None, dtype=np.float32,
                 rpy_dtype=np.float32):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split(':')
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0].strip(),
                                  [float(i) for i
                                   in pair[1].strip().strip('[').strip(']').split(',')]))
        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self._rpy_dtype = rpy_dtype
        self._mean = mean

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, rpy = self._pairs[i]
        full_path = os.path.join(self._root, path)
        with Image.open(full_path) as f:
            f = f.resize((120,120))
            image = np.asarray(f, dtype=self._dtype)
            image /= 255.0
        rpy = np.array(rpy, dtype=self._rpy_dtype)
        image = image.transpose(2, 0, 1)
        if self._mean is not None:
            image -= self._mean
        return image, rpy
