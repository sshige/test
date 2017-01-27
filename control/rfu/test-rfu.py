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

# G_base = np.linalg.qr(np.linalg.pinv(G))[0].transpose()
# G_null_base = np.linalg.qr(G_null)[0].transpose()[0]

np.c_(np.cross(np.array([1,2,3]), np.identity(3)), np.identity(3))

p1 = np.array([-1.0,0.0,3.0])
p2 = np.array([2.0,0.0,3.0])
G=np.r_[np.c_[np.identity(3), np.identity(3)],np.c_[np.cross(p1, np.identity(3)), np.cross(p2, np.identity(3))]]

G=np.r_[np.c_[np.identity(3), np.zeros((3,3)), np.identity(3), np.zeros((3,3))],np.c_[np.cross(p1, np.identity(3)), np.identity(3), np.cross(p2, np.identity(3)), np.identity(3)]]


from __future__ import print_function

for i in range(G_null.shape[0]):
    print('  \\bm{e_{ex}^1} &=& \\begin{pmatrix} ', end='')
    for j in range(G_null[i].shape[0]):
        if np.abs(G_null[i][j]) < 1e-10:
            print(0, end=' & ')
        else:
            print('{0:.2f}'.format(G_null[i][j]), end=' & ')
    print('\\end{pmatrix}^T\\\\')


for i in range(G_base.shape[0]):
    print('  \\bm{e_{ex}^1} &=& \\begin{pmatrix} ', end='')
    for j in range(G_base[i].shape[0]):
        if np.abs(G_base[i][j]) < 1e-10:
            print(0, end=' & ')
        else:
            print('{0:.2f}'.format(G_base[i][j]), end=' & ')
    print('\\end{pmatrix}^T\\\\')

for i in range(G_null_base[0:6].shape[0]):
    print('  \\bm{e_{in}^1} &=& \\begin{pmatrix} ', end='')
    for j in range(G_null_base[0:6][i].shape[0]):
        if np.abs(G_null_base[0:6][i][j]) < 1e-10:
            print(0, end=' & ')
        else:
            print('{0:.2f}'.format(G_null_base[0:6][i][j]), end=' & ')
    print('\\end{pmatrix}^T\\\\')

