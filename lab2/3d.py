import numpy as np
import matplotlib.pyplot as plt
import methods

learning_rate = 0.3
n_epochs = 100
batch_sizes = [1, 10, 20, 50]

np.random.seed(1212)
X = np.random.rand(200, 2)
y = 0 + 10 * X + np.random.randn(200, 1)
w = methods.stochastic_gradient_descent(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(X[:, 0], X[:, 1], y[:, 0], 'b.')
plt.plot(X[:, 0], X[:, 1], X.dot(w)[:, 0], 'r-')
plt.title('sgd')
plt.show()

