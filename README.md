# 02-Data-Science-My-Linear-Regression

<div class="row">
<div class="col tab-content">
<div class="tab-pane active show" id="subject" role="tabpanel">
<div class="row">
<div class="col-md-12 col-xl-12">
<div class="markdown-body">
<p class="text-muted m-b-15">
</p><h1>My Linear Regression</h1>
<p>Remember to git add &amp;&amp; git commit &amp;&amp; git push each exercise!</p>
<p>We will execute your function with our test(s), please DO NOT PROVIDE ANY TEST(S) in your file</p>
<p>For each exercise, you will have to create a folder and in this folder, you will have additional files that contain your work. Folder names are provided at the beginning of each exercise under <code>submit directory</code> and specific file names for each exercise are also provided at the beginning of each exercise under <code>submit file(s)</code>.</p>
<hr>
<table>
<thead>
<tr>
<th>My Linear Regression</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Submit directory</td>
<td>.</td>
</tr>
<tr>
<td>Submit file</td>
<td>my_linear_regression</td>
</tr>
</tbody>
</table>
<h3>Description</h3>
<h2>Subject</h2>
<p>Getting and analysing existing data is the very first task of a data scientist.
The next step is to find tendencies and to generalize.</p>
<p>For example, let's say we want to know what a cat is. We can learn by heart some pictures of cats and then classify as cat animals that are similar to the pictures.
We then need a way to "measure" similarity. This is called instance-based learning.</p>
<p>Another way of generalizing is by creating a model from the existing examples and make prediction based on that model.</p>
<p>For instance, let's say we want to analyze the relation between two attributes and plot one against the other:</p>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/linear_points_nude.png" width="400">
<p>We clearly see a trend here, eventhough the data is quite noisy, it looks like the feature 2 goes up linearly as the feature 1 increases.
So in a model selection step, we can decide to go for a linear model.</p>
<p>feature_2 = θ<sub>0</sub> + θ<sub>1</sub> . feature_1</p>
<p>This model has two parameters, θ<sub>0</sub> and θ<sub>1</sub>. After choosing the right values for them, we can make our model represent a linear function matching the data:</p>
<img src="https://storage.googleapis.com/qwasar-public/track-ds/linear_points_regressed.png" width="400">
<p>Everything stands in "choosing the right values". The "right values" are those for which our model performs "best".
We then need to define a performance measure (how well the model performs) or a cost function (how bad the model performs).</p>
<p>These kind of problems and models are called <strong>Linear Regression</strong>.
The goal of this journey is to explore linear and logistic regressions.</p>
<h2>Introduction</h2>
<p>A linear model makes predictions by computing a weighted sum of the features (plus a constant term called the bias term):</p>
<p>y = h<sub>θ</sub>(x) = θ<sup>T</sup>·<strong>x</strong> = θ<sub>n</sub>x<sub>n</sub> + ... + θ<sub>2</sub>x<sub>2</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>0</sub></p>
<p>• y is the predicted value.
• n is the number of features.
• x<sub>i</sub> is the i<sup>th</sup> feature value (with x<sub>0</sub> always equals to 1).
• θ<sub>j</sub> is the j<sup>th</sup> model feature weight (including the bias term θ<sub>0</sub>).
• · is the dot product.
• h<sub>θ</sub> is called the hypothesis function indexed by θ.</p>
<p>→ <strong>Write the linear hypothesis function.</strong></p>
<pre class=" language-plain"><code class=" language-plain">def h(x, theta):
    ...
</code></pre>
<p>Now that we have our linear regression model, we need to define a cost function to train it, i.e measure how well the model performs and fits the data.
One of the most commonly used function is the Root Mean Squared Error (RMSE). As it is a <strong>cost</strong> function, we will need to optimize it and find the value
of theta which minimizes it.</p>
<p>Since the sqrt function is monotonous and increasing, we can minimize the square of RMSE, the Mean Square Error (MSE) and it will lead to the same result.</p>
<p>&nbsp;<sub>m</sub>
MSE(X, h<sub>θ</sub>) = &nbsp;&nbsp;<sup>1</sup>⁄<sub>m</sub>&nbsp;∑ (θ<sup>T</sup>·x<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>
        &nbsp;<sup>k=1</sup></p>
<p>• X is a matrix which contains all the feature values. There is one row per instance.
• m is the number of instances.
• x<sub>i</sub> is the feature values vector of the i<sup>th</sup> instance
• y<sub>i</sub> is the label (desired value) of the i<sup>th</sup> instance.</p>
<p>→ <strong>Write the Mean Squared Error function between the predicted values and the labels.</strong></p>
<pre class=" language-plain"><code class=" language-plain">def mean_squared_error(y_predicted, y_label):
    ...
</code></pre>
<p>Now our goal is to minimize this MSE function.</p>
<h2>Closed-Form Solution</h2>
<p>To find the value of θ that minimizes the cost function, we can differentiate the MSE with respect to θ.
It directly gives us the correct θ in what we called the <em>Normal Equation</em>:</p>
<p>θ = (X<sup>T</sup>·X)<sup>-1</sup>·X<sup>T</sup>·y</p>
<p>(<strong>NB:</strong> This requires X<sup>T</sup>X to be inversible).</p>
<p>→ <strong>Write a class LeastSquareRegression to calculate the θ feature weights and make predictions.</strong></p>
<p><strong>hint</strong> <a href="https://numpy.org/doc/stable/reference/routines.linalg.html" target="_blank">Checkout Numpy's linear algebra module</a></p>
<pre class=" language-plain"><code class=" language-plain">class LeastSquaresRegression():
    def __init__(self,):
        self.theta_ = None

    def fit(self, X, y):
        # Calculates theta that minimizes the MSE and updates self.theta_


    def predict(self, X):
        # Make predictions for data X, i.e output y = h(X) (See equation in Introduction)

</code></pre>
<p>Let's now use this class on data we are going to generate.
Here is some code to generate some random points.</p>
<pre class=" language-plain"><code class=" language-plain">import numpy as np
import matplotlib.pyplot as plt

X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)
</code></pre>
<p>→ <strong>Plot these points to get a feel of the distribution</strong>.</p>
<p>As you can see, these points are generated in a linear way with some Gaussian noise.
Before calculating our weights, we will account for the bias term (x<sub>0</sub> = 1).</p>
<p>→ <strong>Write a function which adds one to each instance</strong></p>
<pre class=" language-plain"><code class=" language-plain">def bias_column(X):
    ...

X_new = bias_column(x)

print(X[:5])
print(" ---- ")
print(X_new[:5])
</code></pre>
<p>You should see something similar to:</p>
<pre class=" language-plain"><code class=" language-plain">[[0.91340515]
 [0.14765626]
 [3.75646273]
 [2.23004972]
 [1.94209257]]
 ----
[[1.         0.91340515]
 [1.         0.14765626]
 [1.         3.75646273]
 [1.         2.23004972]
 [1.         1.94209257]]

</code></pre>
<p>→ <strong>Calculate the weights with the LeastSquaresRegression class</strong></p>
<pre class=" language-plain"><code class=" language-plain">model = LeastSquaresRegression()
model.fit(X_new, y)

print(model.theta_)
</code></pre>
<p>→ <strong>Are the values consistent with the generating equation (i.e 10 and 2) ?</strong></p>
<p>Let's see what our model predicts !</p>
<p>→ <strong>Use your model to predict values from X and plot the two set of points superimposed.</strong></p>
<pre class=" language-plain"><code class=" language-plain">y_new = model.predict(X_new)

def my_plot(X, y, y_new):
    ...

my_plot(X, y, y_new)

</code></pre>
<p>You should see something similar to the pictures in the subject introduction.</p>
<p>→ <strong>What is the computational complexity of this method?</strong>
→ <strong>How does the training complexity compare to the predictions complexity?</strong></p>
<h2>Gradient Descent</h2>
<h3>Reminder about Gradient Descent</h3>
<p>As you may have noticed, our MSE cost function is a convex function. This means that to find the minimum, a strategy based on a gradient descent
will always lead us to a global optimum. Remember that the gradient descent moves toward the direction of the steepest slope.</p>
<p>We will write a class to perform the gradient descent optimization.</p>
<pre class=" language-plain"><code class=" language-plain">class GradientDescentOptimizer():

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

</code></pre>
<p>Let's use this optimizer with a simple function: f(x) = 3 + (x - (2&nbsp; 6)<sup>T</sup>)<sup>T</sup> · (x - (2&nbsp; 6)<sup>T</sup>). The input of f is a vector of size 2.</p>
<p>→ <strong>Write the f function</strong></p>
<pre class=" language-plain"><code class=" language-plain">def f(x):
    ...
</code></pre>
<p>→ <strong>Write the fprime function</strong></p>
<pre class=" language-plain"><code class=" language-plain">def fprime(x):
    ...
</code></pre>
<p>→ <strong>Use the the gradient descent optimizer to try to find the best theta value</strong></p>
<pre class=" language-plain"><code class=" language-plain">grad = GradientDescentOptimizer(f, fprime, np.random.normal(size=(2,)), 0.1)
grad.optimize(10)
grad.print_result()
</code></pre>
<p>Don't hesitate to tweak the hyper parameters (the learning rate and the number of iterations).</p>
<p>→ <strong>Plot the function f in 3D</strong></p>
<p>→ <strong>Plot the progression of the gradient by using the history variable inside the class</strong></p>
<p>We should clearly see the gradient moving toward the cavity of the function f !</p>
<p>→ <strong>How does the learning rate and the number of iterations influence the result ?</strong></p>
<p>→ <strong>How to <a href="https://en.wikipedia.org/wiki/Hyperparameter_optimization" target="_blank">tune hyperparameters</a> and choose a good learning rate?</strong></p>
<p>→ <strong>What about the number of iterations? What is the Convergence Rate?</strong></p>
<p>The batch gradient descent is good but suffers from a huge difficulty: it uses the whole training data set, which can be computationally and memory expensive.
The time to train a model can be very long, Thus, a lot of variants of the implementation of the gradient descent exists to try to increase its efficiency.
For example, a more light weight strategy is the <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent" target="_blank">Stochastic Gradient Descent</a>.</p>
<h2>Technical Description</h2>
<p>You will have to implement multiple functions and 2 classes:</p>
<h3>Functions</h3>
<pre class=" language-plain"><code class=" language-plain">- def h(x, theta)
Write the linear hypothesis function. (see above)

- def mean_squared_error(y_pred, y_label)
Write the Mean Squared Error function between the predicted values and the labels.

- def bias_column(x)
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
 ----
[[1.         0.91340515]
 [1.         0.14765626]
 [1.         3.75646273]
 [1.         2.23004972]
 [1.         1.94209257]]
</code></pre>
<h3>Classes</h3>
<pre class=" language-plain"><code class=" language-plain">class LeastSquaresRegression: (see description above)
  def __init__(self, )
  def fit()
  def predict

class GradientDescentOptimizer: (see description above)
  def __init__()
  def step()
  def optimize()
  def getCurrentValue()
</code></pre>

<p></p>
</div>

</div>
</div>
</div>
<div class="tab-pane" id="resources" role="tabpanel">
</div>
</div>
</div>
