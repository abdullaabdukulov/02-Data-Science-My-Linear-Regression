import numpy as np
import matplotlib.pyplot as plt


def h(x, theta):
    return np.dot(x, theta).reshape(-1, 1)

def mean_squared_error(y_pred, y_label):
    return np.mean((y_pred - y_label) ** 2)

def bias_column(X):
    one = np.ones((X.shape[0], 1))
    return np.concatenate((one, X), axis=1)

class LeastSquaresRegression:
    def __init__(self):
        self.theta_ = None

    def fit(self, X, y):
        X_b = bias_column(X)  
        self.theta_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = bias_column(X)  
        return h(X_b, self.theta_)

class GradientDescentOptimizer:
    def __init__(self, f, fprime, start, learning_rate=0.1):
        self.f_ = f  
        self.fprime_ = fprime  
        self.current_ = start  
        self.learning_rate_ = learning_rate  
        self.history_ = [start]  

    def step(self):
        new_value = self.current_ - self.learning_rate_ * self.fprime_(self.current_)
        self.current_ = new_value
        self.history_.append(new_value)

    def optimize(self, iterations=100):
        for _ in range(iterations):
            self.step()

    def getCurrentValue(self):
        return self.current_

    def print_result(self):
        print("Best theta found is", self.current_)
        print("Value of f at this theta: f(theta) =", self.f_(self.current_))
        print("Value of f prime at this theta: f'(theta) =", self.fprime_(self.current_))

def f(x):
    a = np.array([[2],[6]])
    t = lambda x: 3 + np.matmul((x - a).T,(x - a))
    return t(x)
    
def f_prime(x):
    a = np.array([[2],[6]])
    t = lambda x: 2*(x - a)
    return t(x)

def my_plot(X, y, y_new):
    plt.scatter(X, y, color='b')
    plt.plot(X,y_new, color='r', linewidth=3)
    plt.legend(['predicted' , 'true'])
    plt.savefig('true_vs_predicted.png')
