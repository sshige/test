#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MyChain(Chain):
    def __init__(self):
        self.l1 = L.Linear(4,3)
        self.l2 = L.Linear(3,2)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)

if __name__ == '__main__':
    model = MyChain()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
