import numpy as np


def gradient_descent(f, grad, start, eps=0.01, lr=0.2, max_iter=10000):
    x = np.array(start)

    g_count = 0
    f_count = 0
    points = [x]
    for i in range(max_iter):
        x = x - lr * grad(x)
        g_count += 1
        points.append(x)
        if abs(f(points[i]) - f(points[i - 1])) <= eps:
            break
        i += 1
    return np.asarray(points)
