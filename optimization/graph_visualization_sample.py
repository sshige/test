#!/usr/bin/env python
# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt

def visualize_contour(colormesh=False, cmap='jet'):
    x = np.arange(0, 10, 0.05) # x range
    y = np.arange(0, 10, 0.05) # y range

    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)   # 表示する計算式の指定。等高線はZに対して作られる。

    if colormesh == True:
        plt.pcolormesh(X, Y, Z, cmap=cmap)
    else:
        plt.contour(X, Y, Z, 100, cmap=cmap)

    pp=plt.colorbar(orientation="vertical") # カラーバーの表示
    pp.set_label("Label", fontsize=24) #カラーバーのラベル

    plt.xlabel('X', fontsize=24)
    plt.ylabel('Y', fontsize=24)

    plt.pause(-1)
