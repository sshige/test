#!/usr/bin/env python

from sklearn.datasets import fetch_mldata
import cv2
import random

mnist = fetch_mldata('MNIST original', data_home="/tmp/mnist")

idlist = range(len(mnist.values()[0]))
random.shuffle(idlist)
for i in idlist:
    img = mnist.values()[0][i].reshape((28, 28))
    number = mnist.values()[3][i]
    img_large = cv2.resize(img, (280, 280))
    print('draw \"%d\" image' % number)
    cv2.imshow('mnist', img_large)
    input = cv2.waitKey(0)
    if input == ord('q'):
        break
    elif  input == ord('p'):
        print(img)
cv2.destroyAllWindows()
