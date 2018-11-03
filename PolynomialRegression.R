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


#1 Create a linear regressor and provide the training set
linear_reg = lm(formula = Salary ~ .,data = dataset)


#2 Create a Polynomial regressor and provide the training set
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,data = dataset)



#3 Visualizing the linear model
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),colour = 'black') +
  geom_line(aes(x = dataset$Level, y = predict(linear_reg, newdata = dataset)),colour = 'red') +
  ggtitle('Salary VS Level (Linear Regression)')
xlab('Level')
ylab('Salary')

#3 Visualizing the Polynomial model
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),colour = 'black') +
  geom_line(aes(x = X_grid, y = predict(poly_reg, newdata = data.frame(Level = X_grid, 
                                                                       Level2 = x_grid^2,
                                                                       Level3 = x_grid^3,
                                                                       Level4 = x_grid^4))),colour = 'red') +
  ggtitle('Salary VS Level (Polynomial Regression)')
xlab('Level')
ylab('Salary')



# Predicting a new result with Linear Regression
predict(linear_reg, data.frame(Level = 7.5))

# Predicting a new result with Polynomial Regression
predict(poly_reg, data.frame(Level = 7.5,
                             Level2 = 7.5^2,
                             Level3 = 7.5^3,
                             Level4 = 7.5^4))


################################ PolYnomial Regression Completed######################################
