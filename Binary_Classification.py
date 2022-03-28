#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 19:17:11 2022

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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load Dataset
dataframe = pd.read_csv("sonar.csv")
dataset = dataframe.values

# Split data
X = dataset[:,0:60].astype(float)
Y = dataset[:,-1]

# converting class labels into integers
encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)

# baseline model
def create_baseline():
   model = Sequential()
   model.add(Dense(60, input_dim=60, activation='relu'))
   model.add(Dense(90, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   # Compile model
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

# # evaluate model
# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results= cross_val_score(estimator, X, encoder_Y, cv=kfold)
# print((results.mean()*100, results.std()*100))


## Building a pipeline using standardized unit
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoder_Y, cv=kfold)
print((results.mean()*100, results.std()*100))




