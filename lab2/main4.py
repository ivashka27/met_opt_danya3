from math import exp

import matplotlib.pyplot as plt
import numpy as np
from lab1.plot import plotter
import methods

np.random.seed(12)


def generate_points(n, dim, f, rg=1):
    X = rg * np.random.rand(n, dim)
    y = []
    X_err = X + 0.1 * np.random.randn(n, dim)
    i = 0
    for x in X_err:
        y.append(f(x))
        i += 1
    return X, np.asarray(y)


def poly_mse(X, y, dim, w):
    res = 0
    for i in range(len(X)):
        x = 0
        for j in range(dim):
            x += (X[i] ** j) * w[j]
        res += (y[i][0] - x[0]) ** 2
    return res / len(X)


def mse_func(X, y, dim):
    return lambda w: poly_mse(X, y, dim, w)


def grad_calculator(x, func, dim):
    h = 1e-5
    res = []
    for i in range(dim):
        delta = np.zeros(dim)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)


def grad_func(f, dim):
    return lambda x: grad_calculator(x, f, dim)


def gradient_descent(x, y, start, grad, learning_rate=lambda epoch: 0.1, eps=1e-6, max_iter=10000):
    w = np.asarray(start)
    iter = 0
    for epoch in range(max_iter):
        iter += 1
        sum_v = 0
        cnt = 0
        cur_v = grad(w)
        w = w - learning_rate(epoch) * cur_v
        sum_v += abs(cur_v)
        cnt += 1
        if abs(sum_v / cnt) < eps:
            break
    return w, iter


func = lambda x: 2 + 1 * x + 3 * x ** 2 - 3 * x ** 3
n = 100
dim = 1
f_dim = 4
rg = 3
(X, y) = generate_points(n, dim, func, rg)
f = mse_func(X, y, f_dim)
grad = grad_func(f, f_dim)

lr = lambda x: 0.001
start = np.zeros(f_dim)
print("start gr:", grad(start))

(points, iter, _) = methods.gradient_descent(f, grad, start, eps_g=1e-2, learning_rate=lr, max_iter=1000)
w = points[-1]
print("w: {}, iter: {}".format(w, iter))


def multiply_x_w(X, w, dim):
    res = []
    for i in range(len(X)):
        cur = 0
        for j in range(dim):
            cur += (X[i] ** j) * w[j]
        res.append(cur)
    return res


plt.plot(X, y, 'b.')
t1 = np.arange(0.0, rg + 0.01, 0.1)
Y = multiply_x_w(t1, w, f_dim)
print(Y)
plt.plot(t1, Y, '-', color='r', label="??", linewidth=3)
plt.legend()
plt.title('sgd')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
