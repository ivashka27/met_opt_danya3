import numpy as np
import matplotlib.pyplot as plt
import methods

n=100
dim=1
np.random.seed(1212)
X = np.random.rand(n, dim)
y = 5 * X + np.random.randn(n, dim)

learning_rate = methods.const_learning_rate
n_epochs = 150
batch_sizes = [1, 10, 20, 50, 100]

print("SGD:")
plt.figure(figsize=(10, 7))
colors = ['r', 'b', 'y', 'g', 'purple']
for i, batch_size in enumerate(batch_sizes):
    w = methods.stochastic_gradient_descent(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), '-', color=colors[i], label=f'Batch size {batch_size}', alpha=0.7)
plt.legend()
plt.title('sgd')
plt.xlabel('x')
plt.ylabel('y')
plt.show()