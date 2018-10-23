#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:13:08 2018

@author: amal
"""

# step 1 : Import the libraries

import numpy as np
import pandas as pd
import matplotlib as mpl

# step 2 : import the dataset

dataset = pd.read_csv('Data.csv')

#step 3 : classify the data into dependent and independent types

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# step 4 : Dealing with missing data

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,1:3] = imp.fit_transform(X[:,1:3])


# step 5 : dealing with categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X = LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
ohe_X = OneHotEncoder(categorical_features = [0])
X = ohe_X.fit_transform(X).toarray()
le_Y = LabelEncoder()
Y = le_X.fit_transform(Y)



# step 6 : Splitting the dataset into Training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# step 7 : Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

