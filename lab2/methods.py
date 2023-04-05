import numpy as np


def stochastic_gradient_descent(x, y, learning_rate=0.1, n_epochs=50, batch_size=1):
    n_samples, n_features = x.shape
    w = np.random.randn(n_features, 1)
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, n_samples, batch_size):
            x_i = x_shuffled[i:i + batch_size]
            y_i = y_shuffled[i:i + batch_size]
            gradient = x_i.T.dot(x_i.dot(w) - y_i) / batch_size
            w = w - learning_rate * gradient
    return w
