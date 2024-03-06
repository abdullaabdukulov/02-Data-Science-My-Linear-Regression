# Description

Getting and analysing existing data is the very first task of a data scientist. The next step is to find tendencies and to generalize.

For example, let's say we want to know what a cat is. We can learn by heart some pictures of cats and then classify as cat animals that are similar to the pictures. We then need a way to "measure" similarity. This is called instance-based learning.

Another way of generalizing is by creating a model from the existing examples and make prediction based on that model.

For instance, let's say we want to analyze the relation between two attributes and plot one against the other:

We clearly see a trend here, eventhough the data is quite noisy, it looks like the feature 2 goes up linearly as the feature 1 increases. So in a model selection step, we can decide to go for a linear model.

feature_2 = θ0 + θ1 . feature_1

This model has two parameters, θ0 and θ1. After choosing the right values for them, we can make our model represent a linear function matching the data:

# Task

A linear model makes predictions by computing a weighted sum of the features (plus a constant term called the bias term):

y = hθ(x) = θT·x = θnxn + ... + θ2x2 + θ1x1 + θ0

• y is the predicted value. • n is the number of features. • xi is the ith feature value (with x0 always equals to 1). • θj is the jth model feature weight (including the bias term θ0). • · is the dot product. • hθ is called the hypothesis function indexed by θ.

# Installation
pip install matplotlib
pip install numpy

# Usage
python my_linear_regression.py