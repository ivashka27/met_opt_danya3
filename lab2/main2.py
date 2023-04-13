import matplotlib.pyplot as plt
import numpy as np
import methods
import math
from math import exp
from lab1.plot import plotter


def f(x):
    # return 10 * x[0] ** 2 + x[1] ** 2
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[0] + 12 * x[1] + 7


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)


def print_info(name, start, points, grad_calc, func_calc):
    print("start:", start)
    print("{}:".format(name))
    print(points[-1], f(points[-1]))
    print("gradient calculations:", grad_calc)
    print("function calculations:", func_calc)


def show_plot(name, start, points, grad_calc, func_calc):
    print_info(name, start, points, grad_calc, func_calc)

    plotter.points_over_contour(points, f)
    plt.title(name)
    plt.show()


start = [-10, 2]

(points1, grad_calc1, func_calc1) = methods.sgd_with_momentum(f, grad, start, learning_rate=lambda epoch: 0.5)
show_plot("Momentum", start, points1, grad_calc1, func_calc1)

(points2, grad_calc2, func_calc2) = methods.sgd_nesterov(f, grad, start, learning_rate=lambda epoch: 0.5)
show_plot("Nesterov", start, points2, grad_calc2, func_calc2)

(points2, grad_calc2, func_calc2) = methods.sgd_adagrad(f, grad, start, learning_rate=lambda epoch: 0.5)
show_plot("AdaGrad", start, points2, grad_calc2, func_calc2)

(points3, grad_calc3, func_calc3) = methods.sgd_rmsprop(f, grad, start, learning_rate=lambda epoch: 0.5)
show_plot("RMSprop", start, points3, grad_calc3, func_calc3)

(points4, grad_calc4, func_calc4) = methods.sgd_adam(f, grad, start, learning_rate=lambda epoch: 0.5)
show_plot("Adam", start, points4, grad_calc4, func_calc4)
