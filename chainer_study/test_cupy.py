#!/usr/bin/env python

import numpy
import cupy
import time

def test(use_cupy = False):
    mat_size = 10000
    if use_cupy:
        rand_mat = cupy.random.randn(mat_size, mat_size)
        ret = cupy.identity(mat_size)
    else:
        rand_mat = numpy.random.randn(mat_size, mat_size)
        ret = numpy.identity(mat_size)
    start = time.time()
    for i in range(2):
        ret = ret.dot(rand_mat)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

if __name__ == '__main__':
    for i in range(3):
        print ("without GPU")
        test(use_cupy = False)
        print ("with GPU")
        with cupy.cuda.Device(0):
            test(use_cupy = True)
