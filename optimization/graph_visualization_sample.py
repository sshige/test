#!/usr/bin/env python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt

def visualize_contour_2d(
        x_range=np.arange(-10, 10, 0.05),
        y_range=np.arange(-10, 10, 0.05),
        func_2d=(lambda x,y: np.sin(x) + np.cos(y)),
        colormesh=False,
        cmap='jet'):
    X, Y = np.meshgrid(x_range, y_range)
    Z = func_2d(X, Y)

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

    plt.pause(-1)
