import math

import numpy as np


def gradient_descent_with_direction(x, y, function, learning_rate=0.1, n_epochs=50, batch_size=1):
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
            cur_v = function(x_i, y_i, w, prev_v)
            w = w - learning_rate * cur_v
            prev_v = cur_v
    return w


def gradient(x_i, y_i, w, batch_size):
    return x_i.T.dot(x_i.dot(w) - y_i) / batch_size


def stochastic_gradient_descent(x, y, learning_rate=0.1, n_epochs=50, batch_size=1):
    return gradient_descent_with_direction(x, y, lambda x_i, y_i, w, prev_v: gradient(x_i, y_i, w, batch_size),
                                           learning_rate, n_epochs, batch_size)


def sgd_with_momentum(x, y, gamma=0.9, learning_rate=0.1, n_epochs=50, batch_size=1):
    return gradient_descent_with_direction(x, y,
                                           lambda x_i, y_i, w, prev_v:
                                           gamma * prev_v + (1 - gamma) * gradient(x_i, y_i, w, batch_size),
                                           learning_rate, n_epochs, batch_size)


def sgd_nesterov(x, y, gamma=0.9, learning_rate=0.1, n_epochs=50, batch_size=1):
    return gradient_descent_with_direction(x, y,
                                           lambda x_i, y_i, w, prev_v:
                                           gamma * prev_v + (1 - gamma) * gradient(
                                               x_i, y_i, w - learning_rate * gamma * prev_v, batch_size),
                                           learning_rate, n_epochs, batch_size)


def sgd_adagrad(x, y, learning_rate=0.1, n_epochs=50, batch_size=1):
    n_samples, n_features = x.shape
    w = np.random.randn(n_features, 1)
    gr_sum = 0
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            x_i = x_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            gr = gradient(x_i, y_i, w, batch_size)
            gr_sum += np.diag(gr).dot(np.diag(gr))
            w = w - learning_rate * gr / math.sqrt(gr_sum)
    return w


def sgd_rmsprop(x, y, beta=0.99, eps=1e-8, learning_rate=0.1, n_epochs=50, batch_size=1):
    n_samples, n_features = x.shape
    w = np.random.randn(n_features, 1)
    prev_s = 0
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            x_i = x_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            gr = gradient(x_i, y_i, w, batch_size)
            s = beta * prev_s + (1 - beta) * (gr ** 2)
            w = w - learning_rate * gr / np.sqrt(s + eps)
            prev_s = s
    return w


def sgd_adam(x, y, beta1=0.9, beta2=0.99, eps=1e-8, learning_rate=0.1, n_epochs=50, batch_size=1):
    n_samples, n_features = x.shape
    w = np.random.randn(n_features, 1)
    prev_v = 0
    prev_s = 0
    beta1_pw = beta1
    beta2_pw = beta2
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            x_i = x_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            gr = gradient(x_i, y_i, w, batch_size)
            v = beta1 * prev_v + (1 - beta1) * gr
            s = beta2 * prev_s + (1 - beta2) * (gr ** 2)
            v_sm = v / (1 - beta1_pw)
            s_sm = s / (1 - beta2_pw)
            w = w - learning_rate * v_sm / np.sqrt(s_sm + eps)
            beta1_pw *= beta1
            beta2_pw *= beta2
            prev_v = v
            prev_s = s
    return w
