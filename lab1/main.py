import numpy as np
import lab1.method.gradient_descent as gd
import lab1.method.dichotomy_gradient as dg
import lab1.plot.plotter as plotter
import lab1.method.func_generation as fg
import lab1.method.wolfe_gradient as wg

# def f(x):
#     return 10 * x[0] ** 2 + x[1] ** 2

N = 2

matrix = fg.generate_func(N, 10)

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

    print("main", (f(x[:, np.newaxis] + h * np.eye(n)) - f(x[:, np.newaxis] - h * np.eye(n))) / (2 * h))
    print("new", grad_calculator(x, f, n))
    return (f(x[:, np.newaxis] + h * np.eye(n)) - f(x[:, np.newaxis] - h * np.eye(n))) / (2 * h)

def grad_calculator(x, func, n):
    h = 1e-5
    res = []
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        res.append((func(x + delta) - func(x - delta)) / (2 * h))
    return np.asarray(res)


#start = 10 * np.random.rand(1, N)[0]
start = [-1, 1]

print("start:", start)
(points1, g_cnt1, f_cnt1) = gd.gradient_descent(f, grad, start, 1e-6, 0.04, 100000)
print("grad desc")
print(points1[-1], f(points1[-1]))
print("g_cnt:", g_cnt1)
print("f_cnt:", f_cnt1)

(points2, g_cnt2, f_cnt2) = dg.dichotomy_gradient(f, grad, start, 1e-6, 1e-3, 1)
print("dich")
print(points2[-1], f(points2[-1]))
print("g_cnt:", g_cnt2)
print("f_cnt:", f_cnt2)

(points3, g_cnt3, f_cnt3) = wg.wolfe_gradient(f, grad, start, 1e-6)

print("wolfe")
print(points3[-1], f(points3[-1]))
print("g_cnt:", g_cnt3)
print("f_cnt:", f_cnt3)

pl2 = plotter.multiple_points_over_contour(points1, points2, points3, f, 0.04)
pl2.show()


