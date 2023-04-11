import timeit
import methods
import numpy as np


def f(x):
    # return 10 * x[0] ** 2 + x[1] ** 2
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[0] + 12 * x[1] + 7


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)

def wrap():
    methods.sgd_nesterov(f, grad, start, learning_rate=lambda epoch: 0.5)

start = [-1, 2]

execution_time = timeit.timeit(wrap, number=1000)
print(f"Время выполнения: {execution_time:.6f} секунд")
