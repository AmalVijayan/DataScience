######################################Data Preprocessing#####################################

#1 importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

#2 dealing with missing data

# dataset$ = ifelse(is.na(dataset$),
#                   ave(dataset$, FUN = function(x) mean(x,na.rm = TRUE)),
#                   dataset$)


#3 categorical data

# dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'),
#                        labels = c(1,2,3))

#4 Splitting into training and testing sets
# 
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Profit, SplitRatio = 0.8)
# 
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# 
# #5 Feature Scaling
# 
# training_set[] = scale(training_set[])
# test_set[] = scale(test_set[])




################################# Data Preprocessing Completed ############################

################################## Ploynomial Regression ##################################


#1 Create a Support Vector regressor and provide the training set
#install.packages('e1071')
library(e1071)
svr_regressor = svm(formula = Salary ~ ., data = dataset, type = 'eps-regression')


#2 Predicting a new result with Support Vector Regression
y_pred = predict(svr_regressor, data.frame(Level = 6.5))


#3 Visualizing the Support Vector Regression model
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),colour = 'black') +
  geom_line(aes(x = X_grid, y = predict(svr_regressor, newdata = data.frame(Level = X_grid))),
            colour = 'red' ) +
  ggtitle('Salary VS Level (Support Vector Regression)')
xlab('Level')
ylab('Salary')






################################ Support Vector Regression Completed######################################
