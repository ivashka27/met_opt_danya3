import methods
import numpy as np
from memory_profiler import memory_usage


def f(x):
    # return 10 * x[0] ** 2 + x[1] ** 2
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[0] + 12 * x[1] + 7


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)


start = [-1, 2]

args = (f, grad, start)
kwargs = {'learning_rate': lambda epoch: 0.5,
          'trajectory': False}


if __name__ == '__main__':
    memory_used = memory_usage((methods.sgd_adam, args, kwargs))
    print(memory_used)
    print(f"Memory used: {max(memory_used)} MB")
