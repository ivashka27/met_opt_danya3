import time
import methods
import numpy as np
import psutil


def f(x):
    # return 10 * x[0] ** 2 + x[1] ** 2
    return 4 * x[0] ** 2 + 9 * x[1] ** 2 - 4 * x[0] * x[1] - 2 * x[0] + 12 * x[1] + 7


def grad(x):
    h = 1e-5
    return (f(x[:, np.newaxis] + h * np.eye(2)) - f(x[:, np.newaxis] - h * np.eye(2))) / (2 * h)

start = [-1, 2]
process = psutil.Process()
avr = 0
iter = 10
for i in range(iter):
    start_cpu = process.cpu_percent()
    methods.sgd_adam(f, grad, start, learning_rate=lambda epoch: 0.5)
    end_cpu = process.cpu_percent()
    avr += end_cpu - start_cpu
    time.sleep(0.1)

print(avr / iter)
