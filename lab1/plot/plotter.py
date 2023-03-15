import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from random import uniform


def generate_n_dots(n, dimensions=2):
    result = []
    for _ in range(n):
        i = []
        for _ in range(dimensions):
            i.append(uniform(-1, 1))
        result.append(i)
    return result


def points_over_function(points, f):
    t = np.linspace(-5, 5, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax1.plot(f(points))
    ax2.plot(t, f(t))
    ax2.plot(points, f(points), 'o-')
    plt.show()


def three_dim_plot(f, sz=2):
    t = np.linspace(-sz, sz, 100)
    X, Y = np.meshgrid(t, t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, f(np.stack((X, Y))))
    plt.show()


def points_over_contour(points, f, name="line", levels=30):
    a = max(-np.min(points), np.max(points)) + 0.1
    t = np.linspace(-a, a, 100)
    X, Y = np.meshgrid(t, t)
    fig, ax = plt.subplots()
    ax.contour(X, Y, f(np.stack((X, Y))), levels=levels)
    l, = ax.plot(points[:, 0], points[:, 1], 'o-', markersize=5)
    ax.plot(points[-1, 0], points[-1, 1], 'x', markersize=10)
    return ax, l


def multiple_points_over_contour(points1, points2, points3, f, lr, name1="gradient descent", name2="dichotomy",
                                 name3="wolfe", levels=30):
    (ax, l1) = points_over_contour(points1, f, name1, levels)
    l2, = ax.plot(points2[:, 0], points2[:, 1], 'o-', markersize=5, color="r", alpha=0.8)
    ax.plot(points2[-1, 0], points2[-1, 1], 'x', color="r", markersize=10)
    l3, = ax.plot(points3[:, 0], points3[:, 1], 'o-', markersize=5, color="yellowgreen", alpha=0.7)
    ax.plot(points3[-1, 0], points3[-1, 1], 'x', color="yellowgreen", markersize=10)
    ax.legend((l1, l2, l3), (name1, name2, name3), loc='upper right', shadow=True)
    plt.title("start: {}, learning rate: {}".format(points1[0], lr))
    return plt


def plot_by_three_coordinates(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x, y, z)
    # fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    ax.set_xlabel('K')
    ax.set_ylabel('N')
    ax.set_zlabel('T(N, K)')

    fig.tight_layout()

    plt.show()


def plot_by_two_coordinates(x, y, name1:str):
    l1 = plt.plot(x, y)
    plt.legend(l1, "gradient descent", loc='upper right', shadow=True)
    plt.xlabel(name1)
    plt.ylabel("gradient descent calculations")
    plt.title("T(" + name1 + ")")
    return plt


def plot_by_array(p_array, start, name, name1="gradient descent", name2="dichotomy", name3="wolfe"):
    l1, = plt.plot(p_array[0][:, 0], p_array[0][:, 1])
    l2, = plt.plot(p_array[1][:, 0], p_array[1][:, 1], color="r", alpha=0.8)
    l3, = plt.plot(p_array[2][:, 0], p_array[2][:, 1], color="yellowgreen", alpha=0.7)
    plt.legend((l1, l2, l3), (name1, name2, name3), loc='upper right', shadow=True)
    plt.xlabel("scaling")
    plt.ylabel(name)
    plt.title("start: {}".format(start))
    return plt
