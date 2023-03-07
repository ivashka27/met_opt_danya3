import numpy as np


def dichotomy_gradient(f, grad, start, eps_f=0.01, eps_r=0.001, lr=0.2, max_iter=10000):
    x = np.array(start)
    points = [x]

    for i in range(max_iter):
        a = 0
        b = lr
        d = grad(x)
        while abs(b - a) > eps_r:
            if f(x - (c * d + eps_r)) < f(x - (c * d - eps_r)):
                a = c
            else:
                b = c

        x = x - ((a + b) / 2.0) * d
        points.append(x)
        if abs(f(points[i]) - f(points[i - 1])) <= eps_f:
            break
        i += 1
    return np.asarray(points)
