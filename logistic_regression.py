import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def logistic_model(X, Y):
    alpha = 0.1
    n_epoch = 100
    n, m = X.shape
    W = np.zeros(n)
    b = 0
    
    for i in range(n_epoch):
        Z = W.dot(X) + b
        p = sigmoid(Z)
        W = W - alpha * (1/m) * np.sum(X * (p - Y), axis=1)
        b = b - alpha * (1/m) * np.sum(p - Y)
        
        J = -(1/m) * np.sum(Y * np.log(p + 1e-8) + (1 - Y) * np.log(1 - p + 1e-8))
        print('epoch =', i, ', cost =', J)
    
    def model(X):
        return W.dot(X) + b > 0
    
    return model


def random_weights(n):
    W = np.tan(np.random.rand(n) * 2 * np.pi)
    b = np.random.rand() * 10 - 5
    
    return W, b

def random_sample(W, b, m, n):
    X = np.random.rand(n, m) * 20 - 10
    Y = W.dot(X) + b > 0
    
    X += np.random.randn(n, m) * 5
    
    return X, Y

m = 50
n = 2
axis = [-20, 20, -20, 20]

W, b = random_weights(n)
X, Y = random_sample(W, b, m, 2)

x1 = -15
x2 = 15
y1 = -1/W[1] * (W[0] * x1 + b)
y2 = -1/W[1] * (W[0] * x2 + b)

plt.figure('Truth')
plt.plot(X[0,:][Y == 0], X[1,:][Y == 0], 'rx')
plt.plot(X[0,:][Y == 1], X[1,:][Y == 1], 'bo')
plt.plot([x1, x2], [y1, y2], 'k')
plt.axis(axis)
plt.title('Truth')

model = logistic_model(X, Y)

Y_pred = model(X)

plt.figure('Predicted')
plt.plot(X[0,:][Y_pred == 0], X[1,:][Y_pred == 0], 'rx')
plt.plot(X[0,:][Y_pred == 1], X[1,:][Y_pred == 1], 'bo')
plt.axis(axis)
plt.title('Predicted')
