#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 00:09:42 2022

@author: bhargobdeka
"""

# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
## Load Data
dataframe = pd.read_csv("diabetes.csv")
dataset = dataframe.values

scaler = MinMaxScaler()
X = dataset[:,0:8]
X = scaler.fit_transform(X)

Y = dataset[:,-1]



# og_X = scalar.inverse_transform(newX)     transform back to original


split_length = int(0.9*np.size(X,axis=0))

# Xtrain = X[0:split_length,:]
# Ytrain = Y[0:split_length,]
# Xtrain = scaler.fit_transform(Xtrain)

# Xtest = X[split_length-1:-1,:]
# Ytest = Y[split_length-1:-1,]
# Xtest = scaler.fit_transform(Xtest)

## Define Model
def create_model(dropout_rate=0.1, weight_constraint=1):
    model = Sequential()
    model.add(Dense(12, input_dim=8,activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation='relu',kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model

kfold = StratifiedKFold(n_splits=10, shuffle=True)
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# # Grid search values for Epochs and Batch size
# batch_size = [10,50,100]
# epochs     = [50,100,150]
# param_grid = dict(batch_size=batch_size, epochs=epochs)

# # Grid search for the optimizer
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(optimizer=optimizer)

# Grid search for dropout rate and weight constraint
weight_constraint = [1, 2]
dropout_rate      = [0.0, 0.1, 0.2]
param_grid = dict(weight_constraint=weight_constraint, dropout_rate=dropout_rate)

# define the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
grid_result = grid.fit(X, Y)

# summarize best results
print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

# scores = cross_val_score(model, X, Y, cv=kfold)
# print((scores.mean()*100, scores.std()*100))


# ## Fit model

# model.fit(Xtrain, Ytrain, epochs=150, batch_size=100, verbose=0)

# ## Evaluate Model
# _, accuracy = model.evaluate(Xtest,Ytest)

# print('Accuracy: %.2f' % (accuracy*100))

# predictions = (model.predict(X)>0.5).astype(int)

## to do
#cross-validation
