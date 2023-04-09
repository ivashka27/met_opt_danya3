import math
from math import exp
import numpy as np

const_learning_rate = lambda epoch: 0.1
exp_learning_rate = lambda epoch: 0.5 * exp(-0.1 * epoch)
step_learning_rate = lambda epoch: 0.1 if epoch < 50 else 0.01


def gradient_descent_with_direction(x, y, function, learning_rate=lambda epoch: 0.1, n_epochs=50, batch_size=1):
    n_samples, n_features = x.shape
    w = np.random.randn(n_features, 1)
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        prev_v = 0
        for i in range(0, n_samples, batch_size):
            x_i = x_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            cur_v = function(x_i, y_i, w, prev_v, epoch)
            w = w - learning_rate(epoch) * cur_v
            prev_v = cur_v
    return w


def gradient(x_i, y_i, w):
    return x_i.T.dot(x_i.dot(w) - y_i) / len(x_i)


def stochastic_gradient_descent(x, y, learning_rate=const_learning_rate, n_epochs=50, batch_size=1):
    return gradient_descent_with_direction(x, y, lambda x_i, y_i, w, prev_v, epoch: gradient(x_i, y_i, w),
                                           learning_rate, n_epochs, batch_size)


def sgd_with_momentum(f, grad, start, gamma=0.9, eps_g=1e-6, learning_rate=const_learning_rate, max_iter=10000):
    x = np.array(start)

    points = [x]
    prev_v = 0
    for epoch in range(max_iter):
        gr = grad(x)
        v = gamma * prev_v + (1 - gamma) * gr
        prev_v = v
        x = x - learning_rate(epoch) * v
        points.append(x)
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), 0


def sgd_nesterov(f, grad, start, gamma=0.9, eps_g=1e-6, learning_rate=const_learning_rate, max_iter=10000):
    x = np.array(start)

    points = [x]
    prev_v = 0
    for epoch in range(max_iter):
        gr = grad(x - learning_rate(epoch) * gamma * prev_v)
        v = gamma * prev_v + (1 - gamma) * gr
        prev_v = v
        x = x - learning_rate(epoch) * v
        points.append(x)
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), 0


def sgd_adagrad(f, grad, start, eps_g=1e-6, learning_rate=const_learning_rate, max_iter=10000):
    x = np.array(start)

    points = [x]
    gr_sum = 0
    for epoch in range(max_iter):
        gr = grad(x)
        gr_sum += np.diag(gr).dot(np.diag(gr))
        x = x - learning_rate(epoch) * gr / math.sqrt(gr_sum)
        points.append(x)
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), 0


def sgd_rmsprop(f, grad, start, beta=0.99, eps_f=1e-8, eps_g=1e-6, learning_rate=const_learning_rate, max_iter=10000):
    x = np.array(start)

    points = [x]
    prev_s = 0
    for epoch in range(max_iter):
        gr = grad(x)
        s = beta * prev_s + (1 - beta) * (gr ** 2)
        x = x - learning_rate(epoch) * gr / np.sqrt(s + eps_f)
        points.append(x)
        prev_s = s
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), 0


def sgd_adam(f, grad, start, beta1=0.9, beta2=0.99, eps_f=1e-8, eps_g=1e-6, learning_rate=const_learning_rate,
             max_iter=10000):
    x = np.array(start)

    points = [x]
    prev_v = 0
    prev_s = 0
    beta1_pw = beta1
    beta2_pw = beta2
    for epoch in range(max_iter):
        gr = grad(x)
        v = beta1 * prev_v + (1 - beta1) * gr
        s = beta2 * prev_s + (1 - beta2) * (gr ** 2)
        v_sm = v / (1 - beta1_pw)
        s_sm = s / (1 - beta2_pw)
        x = x - learning_rate(epoch) * v_sm / np.sqrt(s_sm + eps_f)
        points.append(x)
        beta1_pw *= beta1
        beta2_pw *= beta2
        prev_v = v
        prev_s = s
        if np.linalg.norm(gr) < eps_g:
            break
    return np.asarray(points), len(points), 0
