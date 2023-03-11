import numpy as np
import matplotlib.pyplot as plt


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


def points_over_contour(points, f, levels=30):
    a = max(-np.min(points), np.max(points)) + 0.1
    t = np.linspace(-a, a, 100)
    X, Y = np.meshgrid(t, t)
    fig, ax = plt.subplots()
    ax.contour(X, Y, f(np.stack((X, Y))), levels=levels)
    ax.plot(points[:, 0], points[:, 1], 'o-')
    ax.plot(points[-1, 0], points[-1, 1], 'x', markersize=10)
    return plt


def multiple_points_over_contour(points1, points2, f, levels=30):
    ax = points_over_contour(points1, f)
    ax.plot(points2[:, 0], points2[:, 1], 'o-', color="r")
    ax.plot(points2[-1, 0], points2[-1, 1], 'x', color="r", markersize=10)
    plt.title("start: {}".format(points1[0]))
    return plt
