import numpy as np
import lab1.method.gradient_descent as gd
import lab1.method.dichotomy_gradient as dg
import lab1.plot.plotter as plotter


def f(x):
    return 10 * (x[0] ** 2 + x[1] ** 2)


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)


points = gd.gradient_descent(f, grad, [-0.7, 1.0], 1e-4, 0.08)
plotter.three_dim_plot(f, 100)
pl1 = plotter.points_over_contour(points, f)
pl1.title("gradient descent")
pl1.show()
print(len(points))
print(points[-1], f(points[-1]))

points2 = dg.dichotomy_gradient(f, grad, [-0.7, 1.0], 1e-4, 1e-3, 0.08)
plotter.points_over_contour(points2, f).show()
print(len(points2))
print(points2[-1], f(points2[-1]))

# points3 = fg.fibonacci_gradient(f, grad, [-0.3, 1], 1e-5, lr=0.01)
# plotter.points_over_contour(points3, f).show()
# print(len(points3))
# print(points3[-1], f(points3[-1]))
