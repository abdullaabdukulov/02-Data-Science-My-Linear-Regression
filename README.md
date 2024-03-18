# Welcome to My Liner Regression
***

## Task
This repository contains Python code for implementing simple linear regression using the 
method of least squares and gradient descent optimization.



## Description
Linear regression is a fundamental statistical method used for modeling the relationship between a 
dependent variable and one or more independent variables. The goal of linear regression is to find 
the best-fitting straight line through the data points. This line can then be used to predict 
the dependent variable for new values of the independent variable.



## Installation
The code provided in this repository consists of two main components:

Least Squares Regression

Implemented in the LeastSquaresRegression class.
Fits a linear model to the data using the method of least squares.
Provides methods for fitting the model to the data (fit) and making predictions (predict).
Gradient Descent Optimization

Implemented in the GradientDescentOptimizer class.
Provides a generic framework for optimizing a function using gradient descent.
Allows customization of the objective function, its gradient, starting point, and learning rate.
Provides methods for taking optimization steps (step), performing the optimization (optimize), and retrieving the current value (getCurrentValue).


## Usage
Running the Code

Ensure that you have Python installed on your system.
Install the required dependencies (NumPy and Matplotlib) using pip install numpy matplotlib.
Execute the script containing the provided code.
Customization

You can customize the data generation process by modifying the parameters of the np.random.rand and np.random.randn functions.
Adjust the number of iterations and learning rate for gradient descent optimization by modifying the respective parameters in the GradientDescentOptimizer instantiation.
Visualization

The code includes a function my_plot for visualizing the generated data points and the linear regression predictions.
You can call this function with appropriate arguments to visualize the results.


