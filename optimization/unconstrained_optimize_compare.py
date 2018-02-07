#!/usr/bin/env python
# coding: UTF-8

# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def quadratic_func(x, *args):
    A,b = args
    return 0.5 * np.inner(x, np.dot(A, x)) - np.inner(b, x)

def quadratic_func_prime(x, *args):
    A,b = args
    return np.dot(A, x) - b

def quadratic_func_prime2(x, *args):
    A,b = args
    return A

def rosenbrock_func(x, *args):
    a,b = args
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_func_prime(x, *args):
    a,b = args
    return np.array([-2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2), 2*b*(x[1] - x[0]**2)])


def visualize_contour_2d(func, x_max, x_min, args):
    x_range = x_max - x_min
    x_max += 2.0 * x_range
    x_min -= 2.0 * x_range

    X_range_array=np.linspace(x_min[0], x_max[0], 100)
    Y_range_array=np.linspace(x_min[1], x_max[1], 100)
    X, Y = np.meshgrid(X_range_array, Y_range_array)
    Z = [[func(np.array([X[i, j], Y[i, j]]), *args) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])]

    # graph
    plt.clf()
    plt.contour(X, Y, Z, 100, cmap='jet')
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=18)

    # color bar
    pp=plt.colorbar(orientation="vertical")
    pp.set_label("Z", fontsize=18)

    plt.pause(0.01)


def steepest_decent(func, x0, jac, args, callback):
    alpha=4e-1
    x_current = x0
    for i in xrange(1000):
        # calculate
        grad = jac(x_current, *args)
        delta_x_update = - alpha * grad
        x_current = x_current + delta_x_update
        # callback
        if callback is not None:
            callback(x_current)
        # check convergence
        if np.linalg.norm(grad) < 1e-2:
            break


def main_rosenbrock(func=rosenbrock_func,
                    func_prime=rosenbrock_func_prime,
                    func_prime2=None):
    a = np.random.rand() * 2
    b = np.random.rand() * 200
    args = (a, b)

    x0 = np.random.rand(2) * 10
    x_correct = np.array([a, a**2])

    main(func, func_prime, func_prime2, x0, x_correct, args)

def main_quadratic(func=quadratic_func,
                   func_prime=quadratic_func_prime,
                   func_prime2=quadratic_func_prime2):
    A = np.random.rand(2, 2)
    A = np.dot(A, A.T) + np.eye(2)
    b = np.random.rand(2)
    args = (A, b)

    x0 = np.random.rand(2)
    x_correct = np.linalg.solve(A, b)

    main(func, func_prime, func_prime2, x0, x_correct, args)


def main(func, func_prime, func_prime2,
         x0, x_correct,
         args):
    x_max = np.maximum(x0, x_correct)
    x_min = np.minimum(x0, x_correct)

    def _callback(x):
        x_history.append(x)

    visualize_contour_2d(func, x_max, x_min, args=args)

    x_history = [x0]
    steepest_decent(func, x0, jac=func_prime, args=args, callback=_callback)
    x_history = np.array(x_history)
    x_history_diff = x_history[1:] - x_history[:-1]
    if np.max(np.abs(x_history)) < 1e6:
        plt.quiver(x_history[:-1, 0], x_history[:-1, 1], x_history_diff[:, 0], x_history_diff[:, 1],
                   angles='xy',scale_units='xy',scale=1, zorder=2,
                   color='black', label='SD')

    x_history = [x0]
    minimize(func, x0, method='BFGS', jac=func_prime, args=args, callback=_callback)
    x_history = np.array(x_history)
    x_history_diff = x_history[1:] - x_history[:-1]
    plt.quiver(x_history[:-1, 0], x_history[:-1, 1], x_history_diff[:, 0], x_history_diff[:, 1],
               angles='xy',scale_units='xy',scale=1, zorder=2,
               color='r', label='BFGS')

    x_history = [x0]
    minimize(func, x0, method='Newton-CG', jac=func_prime, hess=func_prime2, args=args, callback=_callback)
    x_history = np.array(x_history)
    x_history_diff = x_history[1:] - x_history[:-1]
    plt.quiver(x_history[:-1, 0], x_history[:-1, 1], x_history_diff[:, 0], x_history_diff[:, 1],
               angles='xy',scale_units='xy',scale=1, zorder=2,
               color='g', label='Newton-CG')

    x_history = [x0]
    minimize(func, x0, method='CG', jac=func_prime, args=args, callback=_callback)
    x_history = np.array(x_history)
    x_history_diff = x_history[1:] - x_history[:-1]
    plt.quiver(x_history[:-1, 0], x_history[:-1, 1], x_history_diff[:, 0], x_history_diff[:, 1],
               angles='xy',scale_units='xy',scale=1, zorder=2,
               color='b', label='CG')

    x_history = [x0]
    minimize(func, x0, method='Nelder-Mead', args=args, callback=_callback)
    x_history = np.array(x_history)
    x_history_diff = x_history[1:] - x_history[:-1]
    plt.quiver(x_history[:-1, 0], x_history[:-1, 1], x_history_diff[:, 0], x_history_diff[:, 1],
               angles='xy',scale_units='xy',scale=1, zorder=2,
               color='y', label='Nelder-Mead')

    plt.legend(loc='lower left')
    plt.pause(0.01)


# main_rosenbrock()
# main_quadratic()
