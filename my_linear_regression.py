import numpy as np
import statistics
import matplotlib.pyplot as plt
import math

X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)
X_1 = X.reshape(1, -1)[0]
Y_1 = y.reshape(1, -1)[0]


def h(x, theta_):
    return np.dot(x, np.array(theta_)).reshape(-1, 1)


class LeastSquaresRegression():
    def __init__(self):
        self.theta_ = None

    def fit(self, X, y):
        X_mean = statistics.mean(X_1)
        Y_mean = statistics.mean(Y_1)
        all_x = [i - X_mean for i in X_1]
        all_y = [i - Y_mean for i in Y_1]
        all_xy = sum([all_x[i] * all_y[i] for i in range(len(all_x))])
        sum_squared_error_x = sum([i ** 2 for i in all_x])
        sum_squared_error_y = sum([i ** 2 for i in all_y])
        E_sum = math.sqrt(sum_squared_error_x * sum_squared_error_y)
        person_corr_score = all_xy / E_sum
        Sy = math.sqrt((sum_squared_error_y) / (len(Y_1) - 1))
        Sx = math.sqrt((sum_squared_error_x) / (len(X_1) - 1))
        slope_line = person_corr_score * (Sy / Sx)
        Y_intercept = Y_mean - slope_line * X_mean
        self.theta_ = np.array([Y_intercept, slope_line]).reshape(-1, 1)
        return self.theta_

    def predict(self, X):
        return h(X, self.theta_)


def my_plot(X, y, y_new):
    plt.title("LeastSquaresRegression")
    plt.scatter(X, y, c=y, cmap="viridis")
    plt.colorbar()
    plt.plot(X, y_new, "r")
    plt.xlabel("X-axis")
    plt.ylabel("Y-pred")
    plt.grid()


def bias_column(X):
    new = np.ones((len(X_1), 1))
    X = np.append(new, X, axis=1)
    return X


X_new = bias_column(X)
print(X[:5])
print(" ---- ")
print(X_new[:5])
model = LeastSquaresRegression()
model.fit(X, y)
model.predict(X_new)


def mean_squared_error(y_predicted, y_label):
    MSE = sum([(y_label[i] - y_predicted[i]) ** 2 for i in range(len(y_label))]) / len(y_label)
    return MSE


def f(x):
    a = np.array([[2], [6]])
    all = 3 + np.dot((x - a).reshape(1, -1), (x - a))
    return all


def fprime(x):
    a = np.array([[2], [6]])
    return 2 * (x - a).reshape(-1,1)

def test_class_gradient_descent_optimizer(self):
    # Create an instance of the GradientDescentOptimizer class
    # with a function f, its derivative f_prime, an initial value,
    # and a learning rate of 0.1.
    gdo = eg.GradientDescentOptimizer(TestAgent.f, TestAgent.f_prime, np.random.normal(size=(2,)), 0.1)
    # Call the optimize method with an argument of 10 to run the optimizer
    # for 10 iterations.
    gdo.optimize(10)
    # Get the current value of the optimizer using the getCurrentValue method.
    user_values = gdo.getCurrentValue()
    # Assert that the first value is between 1.5 and 1.9,
    # and that the second value is between 5.0 and 5.5.
    self.assertTrue(1.5 < user_values[0])
    self.assertTrue(user_values[0] < 1.9)
    self.assertTrue(5.0 < user_values[1])
    self.assertTrue(user_values[1] < 5.5)



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
        for i in range(iterations):
            self.step()

    def getCurrentValue(self):
        return self.current_

    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))