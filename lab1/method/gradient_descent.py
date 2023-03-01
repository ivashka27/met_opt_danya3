import numpy as np


def gradient_descent(f, grad, n, start, lr=0.2, epoch=30):
    x = np.array(start)

    points = np.zeros((epoch, n))
    points[0] = x
    for i in range(1, epoch):
        x = x - lr * grad(x)
        points[i] = x
    return points
