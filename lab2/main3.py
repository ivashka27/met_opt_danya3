import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import methods
from lab1.plot import plotter

np.random.seed(12)


def generate_points(n, dim, f):
    X = np.random.rand(n, dim)
    y = []
    X_err = X + 0.1 * np.random.randn(n, dim)
    for x in X_err:
        y.append(f(x))
    return X, np.asarray(y)


def mse(X, y, w):
    res = 0
    for i in range(0, len(y)):
        x = 0
        for j in range(0, len(X[i])):
            x += X[i][j] * w[j]
        res += np.square(y[i] - x)
    return res / len(y)


def mse_func(X, y):
    return lambda w: mse(X, y, w)


def grad_calculator(x, func, n):
    h = 1e-5
    res = []
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)


def grad_func(f, n):
    return lambda x: grad_calculator(x, f, n)


n = 1000
dim = 2
(X, y) = generate_points(n, dim, lambda x: 10 * x[0] + 2 * x[1])
f = mse_func(X, y)
grad = grad_func(f, dim)
(points, grad_calc, _) = methods.sgd_with_momentum(f, grad, [1, 1])

print("found w:", points[-1])
print("found value:", f(points[-1]))
print("grad_calc:", grad_calc)
(ax, l) = plotter.points_over_contour(points, f)
plt.show()
