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

print("Momentum:")
for i, batch_size in enumerate(batch_sizes):
    w = methods.sgd_with_momentum(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('momentum')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("Nesterov:")
for i, batch_size in enumerate(batch_sizes):
    w = methods.sgd_nesterov(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('nesterov')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("AdaGrad:")
for i, batch_size in enumerate(batch_sizes):
    w = methods.sgd_adagrad(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('AdaGrad')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("RMSprop:")
for i, batch_size in enumerate(batch_sizes):
    w = methods.sgd_rmsprop(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('RMSprop')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


print("Adam:")
for i, batch_size in enumerate(batch_sizes):
    w = methods.sgd_adam(X, y, learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size)
    print("batch: {}. w: {}".format(batch_size, w))
    plt.plot(X, y, 'b.')
    plt.plot(X, X.dot(w), 'r-', label=f'Batch size {batch_size}')
plt.legend()
plt.title('Adam')
plt.xlabel('x')
plt.ylabel('y')
plt.show()