#!/usr/bin/env python
from __future__ import print_function
import argparse
import code
import numpy
import cupy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import Tkinter
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
import sys


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(784, args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # model.predictor.l1(chainer.Variable(cupy.array([range(784)], dtype=numpy.float32)))
    # model.predictor.l1(chainer.Variable(numpy.array([range(784)], dtype=numpy.float32)))

    # for i in range(10):
    #     print(cupy.argmax(model.predictor(chainer.Variable(cupy.array([test[i][0]], dtype=numpy.float32))).data),
    #           test[i][1])

    return model


class Scribble:
    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        self.canvas.create_oval(self.sx, self.sy, event.x, event.y,
                                outline = "black",
                                width = 30)

    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                                fill = "black",
                                width = 30)
        self.sx = event.x
        self.sy = event.y

    def on_pushed_quit(self):
        sys.exit(0)

    def on_pushed_load(self):
        # save temporary eps
        tmp_file_name = "/tmp/tmp.eps"
        self.canvas.postscript(file=tmp_file_name)

        # load eps and convert for NN input
        color_image = Image.open(tmp_file_name)
        gray_image = ImageOps.grayscale(color_image)
        resized_image = gray_image.resize((28,28))
        # resized_image = resized_image.filter(ImageFilter.GaussianBlur(2))
        resized_image.save("/tmp/tmp.png")
        pixlist = numpy.array([[1.0 - i / 255.0 for i in list(resized_image.getdata())]], dtype=numpy.float32)

        # load eps and convert for NN input
        # import scipy.misc
        # gray = scipy.misc.imread(tmp_file_name, mode='L')   # numpy.ndarray
        # resized_image = scipy.misc.imresize(gray, (28, 28))
        # pixlist = 1. - resized_image.flatten() / 255.
        # pixlist = pixlist.astype(numpy.float32)

        # pass to NN and recognize
        ret_all = model.predictor(chainer.Variable(chainer.cuda.to_gpu(pixlist))).data
        # ret_all = model.predictor(chainer.Variable(cupy.array([pixlist], dtype=numpy.float32))).data
        ret = cupy.argmax(ret_all)
        print('# result : {}'.format(ret)),
        for i in zip(range(10),ret_all.tolist()[0]):
            print('# {} : {}'.format(i[0], i[1]))
        self.result_label.configure(text=str(ret), font=('Helvetica', '12'))
        # import ipdb; ipdb.set_trace()

    def on_pushed_clear(self):
        self.canvas.delete("all")

    def create_window(self):
        window = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(window, bg = "white",
                                     width = 280, height = 280)
        self.canvas.pack()

        quit_button = Tkinter.Button(window, text = "Quit",
                                     command = self.on_pushed_quit)
        quit_button.pack(side = Tkinter.RIGHT)

        load_button = Tkinter.Button(window, text = "Load",
                                     command = self.on_pushed_load)
        load_button.pack(side = Tkinter.RIGHT)

        clear_button = Tkinter.Button(window, text = "Clear",
                                      command = self.on_pushed_clear)
        clear_button.pack(side = Tkinter.RIGHT)

        self.result_label = Tkinter.Label(window, text='', fg='black', bg='white', width=5)
        self.result_label.pack(side = Tkinter.LEFT)

        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

        return window;

    def __init__(self, model):
        self.window = self.create_window();
        self.model = model

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    model = main()
    Scribble(model).run()
