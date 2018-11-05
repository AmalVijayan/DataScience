#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:27:33 2018

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

######################### Decision Tree Regression #########################

#1 Create a regressor and provide the dataset
from sklearn.tree import DecisionTreeRegressor
decisionTree_regressor = DecisionTreeRegressor(criterion = 'mse')
decisionTree_regressor.fit(X,Y)



#2 Predict the salary of a level 6.5 employee

y_pred = decisionTree_regressor.predict(6.5)


#3 Visualizing the Decision Tree Regression
x_grid = np.arange(min(X),max(X),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, Y, color = "red")
plt.plot(x_grid, decisionTree_regressor.predict(x_grid), color = "blue")
plt.title('Salary vs Levels (Decision Tree Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()