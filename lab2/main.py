import numpy as np
import matplotlib.pyplot as plt
import methods

n = 300
dim = 1
np.random.seed(1212)
X = 5 * np.random.rand(n, dim)
y = 6 * X + 1 * np.random.randn(n, dim)

learning_rate = methods.const_learning_rate
batch_sizes = [1, 50, 100, 200, 300]

print("SGD:")
colors = ['r', 'orange', 'y', 'g', 'violet']
for i, batch_size in enumerate(batch_sizes):
    w, iters = methods.stochastic_gradient_descent(X, y, learning_rate=learning_rate, batch_size=batch_size)
    print("batch: {}. w: {}. iters: {}".format(batch_size, w, iters))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), '-', color=colors[i], label=f'Batch size {batch_size}', linewidth=3)
    plt.legend()
    plt.title('sgd')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

