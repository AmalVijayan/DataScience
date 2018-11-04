#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:35:26 2018

@author: amal
"""


########################### Data Preprocessing ###################################

#1 Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 importing the data set

dataset = pd.read_csv('Position_Salaries.csv')


#3 classifying dependent and independent variables

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

"""
#4 dealing with missing values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,:-1] = imp.fit_transform(X[:,:-1])
"""

"""
#5 dealing with categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X = LabelEncoder()
X[:,-1] = label_X.fit_transform(X[:,-1])
ohe_X = OneHotEncoder(categorical_features = [3])
X = ohe_X.fit_transform(X).toarray()
"""

"""Avoiding the Dummy variable trap """
#X = X[:,1:]  # some libraries require it to be taken care manualy

"""
#6 Splitting into training and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
"""

#7 Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = Y.reshape((len(Y), 1)) 
Y = sc_Y.fit_transform(Y)
Y = Y.ravel()



#################Data Preprocessing Ends #################################


####################### SVR ###############################################



#1 Create a Linear Regressor and provide the dataset to learn
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)

#2 Create a Support vector  Regressor and provide the dataset

from sklearn.svm import SVR
svr = SVR(kernel='rbf') 
svr.fit(X, Y)



#Visualizing the linear regression model
plt.scatter(X, Y, color = 'black')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Salary vs Level (Linear Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#Visualizing the Support Vector regression model
plt.scatter(X, Y, color = 'black')
plt.plot(X, svr.predict(X), color = 'blue')
plt.title('Salary vs Level (Support Vector Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#predicting the salary for a level 6.5 employee

#convert 6.5 to an array holding 6.5 as an item
#Scale 6.5
#inverse tranfor to get the unscaled or real value
y_predict = sc_Y.inverse_transform(svr.predict(sc_X.transform(6.5)))
#y_predict2 = sc_Y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]]))))