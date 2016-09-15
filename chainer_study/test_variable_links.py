#!/usr/bin/env python

import numpy
import chainer

if __name__ == '__main__':
    x1 = chainer.Variable(numpy.array([[5]], dtype=numpy.float32))
    func1 = chainer.links.Linear(1,1)
    y1 = func1(x1)
    print(x1.data, y1.data)

    x2 = chainer.Variable(numpy.array([[5], [10]], dtype=numpy.float32))
    func2 = chainer.links.Linear(1,1)
    y2 = func2(x2)
    print(x2.data, y2.data)

    x4 = chainer.Variable(numpy.array([[5, 10], [-5, -10]], dtype=numpy.float32))
    func4 = chainer.links.Linear(2,1)
    y4 = func4(x4)
    print(x4.data, y4.data)

    x5 = chainer.Variable(numpy.array([[5, 10], [-5, -10]], dtype=numpy.float32))
    func5 = chainer.links.Linear(2,2)
    y5 = func5(x5)
    print(x5.data, y5.data)

    x6 = chainer.Variable(numpy.array([range(784)], dtype=numpy.float32))
    func6 = chainer.links.Linear(784,10)
    y6 = func6(x6)
    print(x6.data, y6.data)
