#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, train, n_data, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_data)))
        x = gen(z, test=True)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        plot_x = np.linspace(train.input_start, train.input_end, train.data_length)
        plot_y = x
        plt.figure()
        for i in range(n_data):
            plt.plot(plot_x, plot_y[i])

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plt.savefig(preview_path)
        plt.close()
    return make_image
