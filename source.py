# -*- coding: utf-8 -*-
"""
@author: S.Alireza Moazeni (S.A.M.P.8)

"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.models import model_from_json
import numpy
"""
Whenever we work with machine learning algorithms that use a stochastic process (e.g. random
numbers), it is a good idea to initialize the random number generator with a fixed seed value.
This is so that you can run the same code again and again and get the same result. This is useful
if you need to demonstrate a result, compare algorithms using the same source of randomness
or to debug a part of your code
"""

"""
this parameter detrmines which way of evaluating should use
1: without evaluating   2: automatic evaluating  3:manual evaluating  4: k-fold cross validation 
"""
evaluation_method = 4

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# 1 load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

"""
following code will be used in manual evaluating during learning (case 3 in fitting)
"""
if evaluation_method == 3 :
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

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
if evaluation_method != 4 :
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
    validation_split = this pararmeter determines the percentage of data which will be used for automatic evaluation
    is performed.
    """
    
    """
    you should choose one of the following methods for learning the network
    
    """
    #4_1 without evaluating network in the process of learning
    if evaluation_method == 1:
        model.fit(X, Y, nb_epoch=150, batch_size=10)
    
    #4_2 with automatic evaluating during learning
    elif evaluation_method == 2: 
        model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
    
    #4_3 with manual evaluating during learning
    elif evaluation_method == 3: 
        model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)
    
    # 5 evaluate the model
    """
    in this project we had a same dataset for both training and testing the model,
    this is a simplest way
    ! following code is for the case 1 (without evluating during learning) !
    """
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    """
    save model to disk via json format
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    
    """
    load json and create model
    """
    json_file = open( 'model.json' , 'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss= 'binary_crossentropy' , optimizer= 'rmsprop' , metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

else :
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, init= 'uniform' , activation= 'relu' ))
        model.add(Dense(8, init= 'uniform' , activation= 'relu' ))
        model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))
        # Compile model
        model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
        # Fit the model
        model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    
