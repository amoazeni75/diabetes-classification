# -*- coding: utf-8 -*-
"""
@author: S.Alireza Moazeni (S.A.M.P.8)
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy

"""
Whenever we work with machine learning algorithms that use a stochastic process (e.g. random
numbers), it is a good idea to initialize the random number generator with a fixed seed value.
This is so that you can run the same code again and again and get the same result. This is useful
if you need to demonstrate a result, compare algorithms using the same source of randomness
or to debug a part of your code
"""

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# 1 load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# 2 create model
"""
In this example we will use a fully-connected network structure with three layers
Fully connected layers are defined using the Dense class
We can specify the number of neurons in the layer as the first argument
We use random number generated from uniform distribution between 0 and 0.05
We will use the rectifier (relu) activation function on the first two layers and the sigmoid
activation function in the output layer
We use a sigmoid activation function on the output layer to ensure our
network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to
a hard classification of either class with a default threshold of 0.5
"""
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(8, kernel_initializer= 'uniform', activation= 'relu' ))
model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))

# 3 compile model
"""
We must specify the loss function to use to evaluate a set of weights
the optimizer used to search through diffrent weights for the network
for this case we will use logarithmic loss, which for a binary classification problem
is defined as binary_crossantropy
"""
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


# 4 fit model
"""
Now it is time to execute the model on some data. We can train or fit our model on our 
loaded data by calling the fit() function on the model
epoch = the fixed number that the network will be trained on the training data
batch size = the number of instances that are evaluated before a weight update in the network
is performed.
"""
model.fit(X, Y, nb_epoch=150, batch_size=10)

# 5 evaluate the mode
"""
in this project we had a same dataset for both training and testing the model,
this is a simplest way
"""
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
