import numpy as np
import lab1.method.gradient_descent as gd
import lab1.method.dichotomy_gradient as dg
import lab1.plot.plotter as plotter


def f(x):
    return 10 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)


#start = 10 * np.random.rand(1, 2)[0]
start = [1, -2]
print(start)
points = gd.gradient_descent(f, grad, start, 1e-6, 0.01, 100000)
# print(f(points.T))
plotter.three_dim_plot(f, 100)
pl1 = plotter.points_over_contour(points, f)
pl1.title("gradient descent")
pl1.show()
print(len(points))
print(points[-1], f(points[-1]))

points2 = dg.dichotomy_gradient(f, grad, start, 1e-6, 1e-5, 200)
pl2 = plotter.points_over_contour(points2, f)
pl2.title("dichotomy descent")
pl2.show()
print(len(points2))
print(points2[-1], f(points2[-1]))

# points3 = fg.fibonacci_gradient(f, grad, [-0.3, 1], 1e-5, lr=0.01)
# plotter.points_over_contour(points3, f).show()
# print(len(points3))
# print(points3[-1], f(points3[-1]))
