import numpy as np


def wolfe_conditions(f, grad, x, d, t, c1, c2):
    fx = f(x)
    gx = np.dot(grad(x), d)
    ft = f(x + t * d)
    gxt = np.dot(grad(x + t * d), d)
    armijo = ft <= fx + c1 * t * gx
    curant = np.abs(gxt) <= c2 * np.abs(gx)
    return armijo and curant


def wolfe_gradient(f, grad, start, eps=1e-6, c1=1e-4, c2=0.9, alpha=1, max_iter=10000):
    x = np.array(start)

    points = [x]
    func_calc = 0
    for _ in range(max_iter):
        gr = grad(x)
        t = alpha
        d = -gr
        func_calc += 2
        while not wolfe_conditions(f, grad, x, d, t, c1, c2):
            func_calc += 2
            t = t / 2
        x = x + t * d
        points.append(x)
        if np.linalg.norm(gr) < eps:
            break
    return np.asarray(points), len(points), func_calc
