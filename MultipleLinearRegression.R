

######################################Data Preprocessing#####################################

#1 importing the dataset
dataset = read.csv('50_Startups.csv')

#2 dealing with missing data

# dataset$ = ifelse(is.na(dataset$),
#                   ave(dataset$, FUN = function(x) mean(x,na.rm = TRUE)),
#                   dataset$)


#3 categorical data

dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'),
                  labels = c(1,2,3))

#4 Splitting into training and testing sets

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# 
# #5 Feature Scaling
# 
# training_set[] = scale(training_set[])
# test_set[] = scale(test_set[])




################################# Data Preprocessing Completed ############################


############################## Multiple Linear Regression ###################################

#1 Create a regressor and provide the training set

regressor = lm(Profit ~ .,data = training_set)
#                   or
# regressor = lm(Profit ~ R.D.Spend + Administration + Marketing + State ,data = training_set)                  

#2 predict the result for test set

Y_pred = predict(regressor, newdata = test_set)


#3 Using backward elimination to obtain the highly significant Independent variable

regrassor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regrassor)

#removing the independent variable with highest p_value
regressor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend , data = dataset)
summary(regressor)

#removing the independent variable with highest p_value
regressor = lm(Profit ~ R.D.Spend + Marketing.Spend , data = dataset)
summary(regressor)

#removing the independent variable with highest p_value
regressor = lm(Profit ~ R.D.Spend , data = dataset)
summary(regressor)


########################Multiple Linear Regression Completed#######################################