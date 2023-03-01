import numpy as np


def gradient_descent(f, grad, lr=0.2, epoch=30, start=3):
    x = np.array([start])

    points = np.zeros(epoch)
    points[0] = x
    for i in range(1, epoch):
        x = x - lr * grad(x)
        points[i] = x
    return points
