import numpy as np


def gradient_descent(f, grad, start, eps=0.01, lr=0.2, max_iter=10000):
    x = np.array(start)

    points = [x]
    for i in range(max_iter):
        x = x - lr * grad(x)
        points.append(x)
        if abs(f(points[i]) - f(points[i - 1])) <= eps:
            break
    print("gradient descent:")
    print("gradient:", len(points))
    print("fun_cnt:", 0)
    return np.asarray(points)
