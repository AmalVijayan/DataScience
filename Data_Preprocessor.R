
# step 1 : Import the libraries
# R includes all necessary libraries by default. Specific libraries can be installed when needed

# step 2 : import the dataset

dataset = read.csv('Data.csv')

#step 3 : classify the data into dependent and independent types


# step 4 : Dealing with missing data

dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = 'TRUE')),
                     dataset$Age)


dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = 'TRUE')),
                        dataset$Salary)

# step 5 : dealing with categorical data

dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))


dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No','Yes'),
                           labels = c(0,1))

# step 6 : Splitting the dataset into Training and testing sets

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# step 7 : Feature Scaling

training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])