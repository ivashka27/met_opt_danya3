import numpy as np
import matplotlib.pyplot as plt
import methods

np.random.seed(1212)
X = np.random.rand(200, 1)
y = 0 + 10 * X + np.random.randn(200, 1)

learning_rate = methods.const_learning_rate
n_epochs = 100
batch_sizes = [1, 10, 20, 50]

print("SGD:")
plt.figure(figsize=(10, 7))
for i, batch_size in enumerate(batch_sizes):
    w = methods.stochastic_gradient_descent(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('sgd')
plt.xlabel('x')
plt.ylabel('y')
plt.show()