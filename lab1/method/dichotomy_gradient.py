import numpy as np


def dichotomy_gradient(f, grad, start, eps=0.01, lr=0.2, max_iter=10000):
    x = np.array(start)

    points = [x]
    a = 0
    b = lr
    i = 1
    while i < max_iter:
        d = -grad(x)
        while abs(b - a) > eps:
            c = (a + b) / 2
            if f(x + c * d) < f(x - c * d):
                a = c
            else:
                b = c

        x = x + a * d
        points.append(x)
        if abs(f(points[i]) - f(points[i - 1])) <= eps:
            break
        i += 1
    return np.asarray(points)
