import numpy as np
import matplotlib.pyplot as plt
X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)


def h(x, theta):
    return np.dot(x, theta)

def mean_squared_error(y_pred, y_label):
    return np.mean((y_pred - y_label) ** 2)

class LeastSquaresRegression():
    def __init__(self,):
        self.theta_ = None

    def fit(self, X, y):
        self.theta_ = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.theta_

def bias_column(X):
    return np.c_[np.ones((len(X), 1)), X]

X_new = bias_column(X)

print(X[:5])
print(" ---- ")
print(X_new[:5])

model = LeastSquaresRegression()
model.fit(X_new, y)
print(model.theta_)

class GradientDescentOptimizer():
    def __init__(self, f, fprime, start, learning_rate = 0.1):
        self.f_      = f
        self.fprime_ = fprime
        self.current_ = start
        self.learning_rate_ = learning_rate
        self.history_ = [start]

    def step(self):
        gradient = self.fprime_(self.current_)
        self.current_ -= self.learning_rate_ * gradient
        self.history_.append(self.current_)


    def optimize(self, iterations = 100):
        for _ in range(iterations):
              self.step()

    def getCurrentValue():
        return self.current_


    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))


y_new = model.predict(X_new)
def my_plot(X, y, y_new):

    plt.scatter(X, y)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Randomly Generated Data")
    plt.show()
    plt.scatter(X, y)
    plt.plot(X, y_new, color='red')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Prediction")
    plt.show()

my_plot(X, y, y_new)


