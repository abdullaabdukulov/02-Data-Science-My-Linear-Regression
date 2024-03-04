# Welcome to My Linear Regression Project

## Task
This project requires from us creating two classes and some functions to make linear Regression model.
1. LeastSquaresRegression
    class LeastSquaresRegression():

        def __init__(self,):
            self.theta_ = None
    
        def fit(self, X, y):
            # Calculates theta that minimizes the MSE and updates self.theta_


        def predict(self, X):
            # Make predictions for data X, i.e output y = h(X) (See equation in Introduction)

   2. GradientDescentOptimizer
       
            class GradientDescentOptimizer():

           def __init__(self, f, fprime, start, learning_rate = 0.1):
               self.f_      = f                       # The function
               self.fprime_ = fprime                  # The gradient of f
               self.current_ = start                  # The current point being evaluated
               self.learning_rate_ = learning_rate    # Does this need a comment ?
    
               # Save history as attributes
               self.history_ = [start]
    
           def step(self):
               # Take a gradient descent step
               # 1. Compute the new value and update selt.current_
               # 2. Append the new value to history
               # Does not return anything
    
    
            def optimize(self, iterations = 100):
                # Use the gradient descent to get closer to the minimum:
                # For each iteration, take a gradient step
    
            def getCurrentValue():
              # Getter for current_
    
            def print_result(self):
                print("Best theta found is " + str(self.current_))
                print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
                print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))

## Description
```angular2html
- def h(x, theta)
Write the linear hypothesis function. (see above)



    def mean_squared_error(y_pred, y_label)
    Write the Mean Squared Error function between the predicted values and the labels.



    def bias_column(x)
    Write a function which adds one to each instance.



X_new = bias_column(x)

print(X[:5])
print(" ---- ")
print(X_new[:5])

You should see something similar to:

[[0.91340515]
[0.14765626]
[3.75646273]
[2.23004972]
[1.94209257]]


[[1.         0.91340515]
[1.         0.14765626]
[1.         3.75646273]
[1.         2.23004972]
[1.         1.94209257]]

```

## Installation
For working with this project you need to install required libraries mentioned in requirements.txt Write it in your 
terminal pip install -r requirements.txt

## Usage
For using this project you need to open my_linear_regression.ipynb and run it. It's jupyter notebook file, so open it 
in jupyter notebook
