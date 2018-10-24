#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: amal

#I.PreProcessing


#1 Importing essential libraries (Included by default in R)

#2 import the dataset

dataset = read.csv('Salary_Data.csv')

#3 classify dependent and independent variables


#4 missing values

# dataset$ = ifelse(is.na(dataset$), 
#                   ave(dataset$, FUN = function(x) mean(x, na.rm = TRUE)),
#                   dataset$)

#5 categorical data

# dataset$ = factor(dataset$,
#                   levels = c('')
#                   labels = c('')


#6 training set and testing set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


#7 feature scaling
# training_set[] = scale(training_set[])
# test_set[] = scale(test_set[])


#  ###############PreProcessing Completed ################### #

## II. Simple Linear Regressor 

#1 import SLR library
#included by default

#2 define the Regressor machine and Train the Regressor with training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#3 predict the outcome of test sets
X_test_Pred = predict(regressor, newdata = test_set)

#4 visualizing training set predictions
#install.packages('ggplot2')
library(ggplot2)

ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),colour = 'black') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),colour = 'red') +
  ggtitle('Salary VS Experience in Years (Training Set)')
  xlab('Years of Experience')
  ylab('Salary')

#4 visualizing test set predictions
  ggplot() + 
    geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),colour = 'blue') +
    geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),colour = 'red') +
    ggtitle('Salary VS Experience in Years (Test Set)')
  xlab('Years of Experience')
  ylab('Salary')

# ###############Simple Linear Regressor Completed ####################### #

