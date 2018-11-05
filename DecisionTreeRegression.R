

################################### Data Preprocessing ################################

#1 Import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

################################### Data Preprocessing Completed ################################


################################### Decision Tree Regression ################################

#1 Create a Decision Tree regressor and provide the training set
#install.packages('rpart')
library(rpart)
decisionTree_regressor = rpart(Salary ~ ., dataset, control = rpart.control(minsplit = 1))


#2 Predicting the salary for a 6.5 level employee

y_pred = predict(decisionTree_regressor, data.frame(Level = 6.5))


#3 Visualizing the Decition Tree Regression Results
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(dataset$Level, dataset$Salary),color= 'red') +
  geom_line(aes(x_grid, predict(decisionTree_regressor, data.frame(Level = x_grid)), color = 'black'))+
  ggtitle('Salary vs Level (Decision Tree Regression) ')
xlab('Level')
ylab('Salary')

################################### Decision Tree Regression Completed ################################