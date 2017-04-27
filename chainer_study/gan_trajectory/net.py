#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, test, sigma=0.01):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


class Generator(chainer.Chain):

    def __init__(self, n_hidden, data_length=20, wscale=0.02):
        self.n_hidden = n_hidden
        w = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0=L.Linear(self.n_hidden, 100, initialW=w),
            dc1=L.Linear(100, 1000, initialW=w),
            dc2=L.Linear(1000, 1000, initialW=w),
            dc3=L.Linear(1000, 100, initialW=w),
            dc4=L.Linear(100, data_length, initialW=w),
            bn0=L.BatchNormalization(100),
            bn1=L.BatchNormalization(1000),
            bn2=L.BatchNormalization(1000),
            bn3=L.BatchNormalization(100),
        )

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, z, test=False):
        h = F.relu(self.bn0(self.l0(z), test=test))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        # x = F.sigmoid(self.dc4(h))
        x = self.dc4(h)
        return x


class Discriminator(chainer.Chain):

    def __init__(self, data_length=20, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0=L.Linear(data_length, 100, initialW=w),
            c1=L.Linear(100, 200, initialW=w),
            c2=L.Linear(200, 200, initialW=w),
            c3=L.Linear(200, 100, initialW=w),
            l4=L.Linear(100, 1, initialW=w),
            bn1=L.BatchNormalization(200, use_gamma=False),
            bn2=L.BatchNormalization(200, use_gamma=False),
            bn3=L.BatchNormalization(100, use_gamma=False),
        )

    def __call__(self, x, test=False):
        h = add_noise(x, test=test)
        h = F.leaky_relu(add_noise(self.c0(h), test=test))
        h = F.leaky_relu(add_noise(self.bn1(
            self.c1(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn2(
            self.c2(h), test=test), test=test))
        h = F.leaky_relu(add_noise(self.bn3(
            self.c3(h), test=test), test=test))
        return self.l4(h)
