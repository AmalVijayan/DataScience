#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 07:52:16 2018

@author: amal
"""

"""I.PreProcessing
"""

#1 Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as ptl

#2 import the dataset
dataset = pd.read_csv('Salary_Data.csv')

#3 classify dependent and independent variables
iX = dataset.iloc[:,:-1].values  #independent variable YearsofExperience
dY = dataset.iloc[:,-1].values  #dependent variable salary

# missing values
"""
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
"""
#categorical data
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X = LabelEncoder()
le_X.fit_transform()
ohe_X = OneHotEncoder(categorical_features=[])
ohe_X.fit_transform().toarray()
"""
#training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iX ,dY, test_size = 1/3,random_state = 0) 

#feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit_transform(X_train)
sc_X.transform(X_test)
"""

"""  ###############PreProcessing Completed ################### """

"""# II. Simple Linear Regressor 
"""
#1 import SLR library
from sklearn.linear_model import LinearRegression

#2 Train the Regressor with training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#3 predict the outcome of test sets
X_test_Pred = regressor.predict(X_test)

#4 visualizing training set predictions
ptl.scatter(X_train,Y_train, color = 'black')
ptl.plot(X_train,regressor.predict(X_train),color = 'red')
ptl.title('Experience VS Salary (Training Set)')
ptl.xlabel('Experience in Years')
ptl.ylabel('Salary')
ptl.show()

#4 visualizing test set predictions
ptl.scatter(X_test,Y_test, color = 'black')
ptl.plot(X_train,regressor.predict(X_train),color = 'blue')
ptl.title('Experience VS Salary (Test Set)')
ptl.xlabel('Experience in Years')
ptl.ylabel('Salary')
ptl.show()


""" ###############Simple Linear Regressor Completed ####################### """

