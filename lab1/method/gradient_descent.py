import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad):
    lr = 0.2
    epoch = 30
    x = np.array([1])

    points = np.zeros((epoch))
    points[0] = x
    for i in range(1, epoch):
        x = x - lr * grad(x)
        points[i] = x

    t = np.linspace(-5, 5, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax1.plot(f(points))
    # ax1.grid()
    ax2.plot(t, f(t))
    ax2.plot(points, f(points), '-', 'o', )
    # ax2.contour(X, f(np.stack((X))), levels=np.sort(np.concatenate((f(points.T), np.linspace(-1, 1, 100)))))
    print(points[-1], f(points[-1]))
    plt.show()