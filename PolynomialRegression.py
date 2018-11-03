#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:09:29 2018

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

"""
#7 Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)

"""


#################Data Preprocessing Ends #################################



########################## Polynomial Regression ##################################


#1 Create a linear regressor and provide the dataset
from sklearn.linear_model import LinearRegression 
Lin_regressor = LinearRegression()
Lin_regressor.fit(X, Y)


#2 Create a Polynomial regressor and update the independent variable set

from sklearn.preprocessing import PolynomialFeatures
Poly_reg = PolynomialFeatures(degree = 4)
X_poly = Poly_reg.fit_transform(X)

Lin_regressor2 = LinearRegression()
Lin_regressor2.fit(X_poly, Y)

#Visualizing the linear regression model
plt.scatter(X, Y, color = 'black')
plt.plot(X, Lin_regressor.predict(X), color = 'blue')
plt.title('Salary vs Level (Linear Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial regression model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color = 'black')
plt.plot(X_grid, Lin_regressor2.predict(Poly_reg.fit_transform(X_grid)), color = 'red')
plt.title('Salary vs Level (Polynomial Model)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#Predicted salary of a level  7.5 employee
Lin_regressor2.predict(Poly_reg.fit_transform(7.5))
