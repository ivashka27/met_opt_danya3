import numpy as np


def dichotomy_method(f, a, b):
    epoch = 30
    eps = 0.2
    points = np.zeros(epoch)

    for i in range(epoch):
        x = (a + b) / 2
        f1 = f(x - eps)
        f2 = f(x + eps)
        if f1 < f2:
            b = x
        else:
            a = x
        points[i] = x
    return points
