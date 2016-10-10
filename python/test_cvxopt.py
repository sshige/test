#!/usr/bin/python

from cvxopt import matrix
from cvxopt import solvers
import numpy as np

W = np.array([[2, .5], [.5, 1]])
h = np.array([1.0, 1.0])

A = np.array([1.0, 1.0]).reshape(1,2)
b = np.array([1.0])

C = np.array([[-1.0, 0.0], [0.0, -1.0]])
d = np.array([0.0, 0.0])

print("W:{}, h:{}, A:{}, b:{}, C:{}, d:{}".format(W.shape, h.shape, A.shape, b.shape, C.shape, d.shape))

sol = solvers.qp(matrix(W), matrix(h), matrix(C), matrix(d), matrix(A), matrix(b))

print sol['x']
