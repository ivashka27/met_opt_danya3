import numpy as np
import lab1.method.gradient_descent as gd
import lab1.method.dichotomy_gradient as dg
import lab1.plot.plotter as plotter
import lab1.method.func_generation as fg

# def f(x):
#     return 10 * x[0] ** 2 + x[1] ** 2

N = 2

matrix = fg.generate_func(N, 20)

print(matrix)

def multiply(x, m, n):
    v = np.zeros(n, dtype=object)
    for i in range(n):
        for j in range(n):
            v[i] += x[j] * m[j][i]
    res = 0
    for i in range(n):
        res += v[i] * x[i]
    return res


def f(x):
    n = N
    return multiply(x, matrix, n)


def grad(x):
    h = 1e-5
    n = N
    return (f(x[:, np.newaxis] + h * np.eye(n)) - f(x[:, np.newaxis] - h * np.eye(n))) / (2 * h)


start = 10 * np.random.rand(1, N)[0]

print("start:", start)
points = gd.gradient_descent(f, grad, start, 1e-6, 0.04, 100000)[0]

plotter.three_dim_plot(f, 100)
print(points[-1], f(points[-1]))

points2 = dg.dichotomy_gradient(f, grad, start, 1e-6, 1e-3, 1)[0]
pl2 = plotter.multiple_points_over_contour(points, points2, f, 0.04)
pl2.show()
print(points2[-1], f(points2[-1]))
