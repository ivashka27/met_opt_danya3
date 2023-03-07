import numpy as np


def dichotomy_gradient(f, grad, start, eps_f=0.01, eps_r=0.001, lr=0.2, max_iter=10000):
    x = np.array(start)

    points = [x]
    a = 0
    b = lr

    g_count = 0
    f_count = 0
    for i in range(max_iter):
        d = grad(x)
        g_count += 1
        while abs(b - a) > eps_r:
            c = (a + b) / 2
            f_count += 2
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
