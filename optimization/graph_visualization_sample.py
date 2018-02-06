#!/usr/bin/env python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime


def visualize_contour_2d(
        x_range=np.arange(-10, 10, 0.05),
        y_range=np.arange(-10, 10, 0.05),
        func_2d=(lambda x: np.sin(x[0]) + np.cos(x[1])),
        colormesh=False,
        cmap='jet',
        pause=-1):
    X, Y = np.meshgrid(x_range, y_range)
    Z = func_2d((X, Y))

    # graph
    plt.clf()
    if colormesh == True:
        plt.pcolormesh(X, Y, Z, cmap=cmap)
    else:
        plt.contour(X, Y, Z, 100, cmap=cmap)
    plt.xlabel('X', fontsize=24)
    plt.ylabel('Y', fontsize=24)

    # color bar
    pp=plt.colorbar(orientation="vertical")
    pp.set_label("Z", fontsize=24)

    plt.pause(pause)

def optimize_grad_2d(
        func_2d=(lambda x: np.sin(x[0]) + np.cos(x[1])),
        x_start=np.array([0,0]),
        alpha=1e-1,
        epsilon=1e-3,
        grad_thre=1e-3):
    iteration_count = 0
    x_current = x_start
    visualize_contour_2d(func_2d=func_2d, pause=0.1)
    for i in range(1000):
        iteration_count += 1
        # calculate
        grad = approx_fprime(x_current, func_2d, epsilon)
        delta_x_update = - alpha * grad
        x_new = x_current + delta_x_update
        # check convergence
        if np.linalg.norm(grad) < grad_thre:
            print("detected convergence !")
            break
        # print
        if iteration_count%10 == 0:
            print("{0:>6} iteration,  x_current = [{1[0]:.3f}, {1[1]:.3f}],  \
grad= [{2[0]:.3f}, {2[1]:.3f}]".format(
                iteration_count, x_current, grad))
        # draw arrow
        plt.quiver(x_current[0], x_current[1], delta_x_update[0], delta_x_update[1],
                   angles='xy',scale_units='xy',scale=1, zorder=2)
        if iteration_count%10 == 0:
            plt.pause(0.01)
        # update
        x_current = x_new
    # print result
    print("==============================")
    print("iteration_count: {0}, convergence x: [{1[0]:.3f}, {1[1]:.3f}]".format(
        iteration_count, x_current))
    print("==============================")
