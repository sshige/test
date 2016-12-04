#! /usr/bin/env python

import numpy as np


def get_G_matrix_for_2D_manip(
        p_l = np.array([-0.1, 0.5]),
        p_r = np.array([0.3, 0.25])
        ):
    G = np.matrix([[1, 0, 1, 0], [0, 1, 0,1], [p_l[1], -p_l[0], p_r[1], -p_r[0]]],
                  dtype=np.float32)
    return G

def get_G_null_matrix(G):
    G_null = np.identity(G.shape[1]) - np.linalg.pinv(G).dot(G)
    return G_null

def check_linear_dependent(X):
    for i in range(X.shape[0]):
        print(np.ravel(np.array(X[i] / X[0])))

# m = 1.0
# p_g = np.array([0.0, 0.4])

GG=np.linalg.qr(np.linalg.pinv(G))[0].transpose()
GnullG=np.linalg.qr(G_null)[0].transpose()[0]
