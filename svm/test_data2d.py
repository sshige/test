#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pylab
from svm_classifier import SVM_classifier

class GenerateData2D:
    def __init__(self, data_num=500, error_rate=0.1):
        self.xrange = [-10.0, 10.0]
        self.yrange = [-3.0, 3.0]
        self.dataset = []
        for i in range(data_num):
            self.dataset.append(self.generate_data_one(error_rate=error_rate))

    def bound_func(self, x):
        return np.sin(0.5 * x) + np.sin(1.0 * x) - np.sin(2.0 * x) + 0.25 * x

    def generate_data_one(self, error_rate=0.0):
        x = (self.xrange[1] - self.xrange[0]) * np.random.rand() + self.xrange[0]
        y = (self.yrange[1] - self.yrange[0]) * np.random.rand() + self.yrange[0]
        l = self.get_label_from_xy(x, y)
        if np.random.rand() < error_rate:
            l = -1.0 if l == 1.0 else 1.0
        return [[x, y], l]

    def get_label_from_xy(self, x, y):
        y_bound = self.bound_func(x)
        return 1.0 if y >= y_bound else -1.0

    def plot(self):
        x_bound = np.linspace(self.xrange[0], self.xrange[1], 1000, endpoint=True)
        y_bound = self.bound_func(x_bound)
        plt.plot(x_bound, y_bound, color='black')
        ds1 = filter((lambda x: x[1] == 1), self.dataset)
        x1 = [d[0][0] for d in ds1]
        y1 = [d[0][1] for d in ds1]
        ds2 = filter((lambda x: x[1] == -1), self.dataset)
        x2 = [d[0][0] for d in ds2]
        y2 = [d[0][1] for d in ds2]
        plt.scatter(x1, y1, color='blue')
        plt.scatter(x2, y2, color='red')
        plt.pause(0.1)

    def visualize_region(self, svmc, division_num=25):
        xgrid = np.linspace(self.xrange[0], self.xrange[1], division_num)
        ygrid = np.linspace(self.yrange[0], self.yrange[1], division_num)
        xwidth = (self.xrange[1] - self.xrange[0]) / division_num
        ywidth = (self.yrange[1] - self.yrange[0]) / division_num
        for i in range(division_num - 1):
            for j in range(division_num - 1):
                x = 0.5 * (xgrid[i] + xgrid[i + 1])
                y = 0.5 * (ygrid[j] + ygrid[j + 1])
                if svmc.test([x, y]) >= 0:
                    facecolor = 'red'
                else:
                    facecolor = 'green'
                rect = pylab.Rectangle(
                    (xgrid[i], ygrid[j]), xwidth, ywidth,
                    facecolor=facecolor, alpha=0.2, linewidth='0')
                pylab.gca().add_patch(rect)
        plt.pause(0.1)

if __name__ == '__main__':
    generate_data = GenerateData2D()
    ds = generate_data.dataset

    svmc = SVM_classifier(ds)
    svmc.optimize()
    svmc.validate_dataset(ds[::10])

    generate_data.plot()
    generate_data.visualize_region(svmc)
    # plt.show()
