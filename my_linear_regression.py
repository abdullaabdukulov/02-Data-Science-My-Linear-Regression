import numpy as np
import matplotlib.pyplot as plt


def h(x, theta):
    return np.dot(x, theta)

def mean_squared_error(y_pred, y_label):
    return np.mean((y_pred - y_label)**2)

def bias_column(x):
    return np.c_[np.ones((x.shape[0], 1)), x]


class LeastSquaresRegression:
    def __init__(self):
        self.theta_ = None

    def fit(self, X, y):
        X_b = bias_column(X)
        self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = bias_column(X)
        return X_b.dot(self.theta_)

class GradientDescentOptimizer:
    def __init__(self, f, fprime, start, learning_rate=0.1):
        self.f_ = f
        self.fprime_ = fprime
        self.current_ = start
        self.learning_rate_ = learning_rate
        self.history_ = [start]

    def step(self):
        gradient = self.fprime_(self.current_)
        self.current_ -= self.learning_rate_ * gradient
        self.history_.append(self.current_)

    def optimize(self, iterations=100):
        for _ in range(iterations):
            self.step()

    def getCurrentValue(self):
        return self.current_

    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))


X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)


plt.scatter(X, y)
plt.xlabel('Feature 1')
plt.ylabel('Output')
plt.title('Generated Data')
plt.show()


X_new = bias_column(X)


model = LeastSquaresRegression()
model.fit(X, y)
print("Theta values:", model.theta_)


y_new = model.predict(X)


plt.scatter(X, y, color='blue', label='Original Data', alpha=0.5)
plt.plot(X, y_new, color='red', label='Predicted Line')
plt.xlabel('Feature 1')
plt.ylabel('Output')
plt.legend()
plt.title('Linear Regression')
plt.show()
