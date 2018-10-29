#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:32:07 2018

@author: amal
"""

########################### Data Preprocessing ###################################

#1 Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 importing the data set

dataset = pd.read_csv('50_Startups.csv')


#3 classifying dependent and independent variables

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

"""
#4 dealing with missing values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
X[:,:-1] = imp.fit_transform(X[:,:-1])
"""


#5 dealing with categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X = LabelEncoder()
X[:,-1] = label_X.fit_transform(X[:,-1])
ohe_X = OneHotEncoder(categorical_features = [3])
X = ohe_X.fit_transform(X).toarray()

"""Avoiding the Dummy variable trap """
X = X[:,1:]  # some libraries require it to be taken care manualy

#6 Splitting into training and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

"""
#7 Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)

"""


#################Data Preprocessing Ends #################################



""" Multiple Linear regression """

#1 Creating the Regressor machine and providing the training sets

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#2 Predicting the result for the test set

Y_pred = regressor.predict(X_test)


#3 performing backward elimination to find X_opt
import statsmodels.formula.api as sm
# add a column of ones to match the regression formulae
X = np.append(arr = np.ones(shape = (50,1)).astype(int),values = X, axis = 1)


"""
# Manual Backward Elimination
X_opt = X[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

#removing X[] with greatest p_value 

X_opt = X[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

#removing X[] with greatest p_value

X_opt = X[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

#removing X[] with greatest p_value

X_opt = X[:, [0,3,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

#removing X[] with greatest p_value

X_opt = X[:, [0,3]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()
"""


# Automating Backward Elimination

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x  
 
SL = 0.05  # significance level
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # optimum array of ind. variables
X_Modeled = backwardElimination(X_opt, SL) # the independepend variable array with highest significance  