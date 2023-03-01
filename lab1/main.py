import lab1.method.gradient_descent as gd
import numpy as np

def f(x):
    return 2 * (x ** 2) + 2 * x + 1


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(1)) - f(x[:, np.newaxis] - h * np.eye(1))) / (2 * h)


gd.gradient_descent(f, grad)