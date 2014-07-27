# Preditive model for Weight Lifting Style using Accelerometer dataset
library(caret)
library(randomForest)

## Data Loading
#The pml-training was downloaded from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv and saved in the project directory
#The pml-testing was downloaded from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv and saved in the project directory
##

## read the training dataset 
raw_dataset = read.csv('pml-training.csv')
## read the validation dataset 
pml_testing = read.csv('pml-testing.csv')


## Data Preparation for modeling
#Partion the raw_dataset into two parts - one for training the model and the other for testing model

#summary(raw_dataset)
set.seed(5678)
trainIndex = createDataPartition(raw_dataset$classe, list=FALSE, p=0.7)
train_set = raw_dataset[trainIndex,]
test_set  = raw_dataset[-trainIndex,]
# since most of the stistic columns have no data, let us get rid of them
#goodCols = !sapply(strsplit(names(train_set),"_"), function(x) x[1]) %in% c("kurtosis","skewness","max","min","amplitude","var","avg","stddev")
#train_set = train_set[,goodCols]
#test_set = test_set[,goodCols]
#summary(train_set)

#Remove all indicators with near zero variance as they are not useful in modeling. After that remove any columns that are not numeric, but keep the classe varaible as this is the outcome variable 

nzerovar = nearZeroVar(train_set)
train_set = train_set[-nzerovar]
test_set = test_set[-nzerovar]
pml_testing = pml_testing[-nzerovar]

numIndex = which(lapply(train_set,class) %in% c('numeric'))
train_set1 = train_set[,numIndex]
test_set1 = test_set[,numIndex]
pml_test1 = pml_testing[,numIndex]

#Now, impute any missing values in the training data set

preModel = preProcess(train_set1, method=c('knnImpute'))
ptrain_set = predict(preModel, train_set1)
ptest_set  = predict(preModel, test_set1)
pml_test_set = predict(preModel, pml_test1)

# add the classe column
ptrain_set = cbind(train_set$classe, ptrain_set)
ptest_set  = cbind(test_set$classe, ptest_set)
# need to fix the column name
names(ptrain_set)[1] = "classe"
names(ptest_set)[1] = "classe"

# Model Building

#Using the "caret" package, build a random forest model, since computationally intensive will use optimized parameters
#Then check the accuracy of the model for in-sample data

RFmodel = randomForest(classe ~ ., ptrain_set, ntree=500, mtry=32)

# check accuracy of model for in-sample data
train_pred = predict(RFmodel, ptrain_set)
confusionMatrix(train_pred, ptrain_set$classe)

#As we can see that the accuracy for the model is perfect for training data, which could sometimes indicate an overfit based on the training dataset. This can be verified by checking the accuracy for test data

test_pred = predict(RFmodel, ptest_set)
confusionMatrix(test_pred, ptest_set$classe)

## Validation of the Model

#from the results - the accuracy of the model is close to 99%, let us use the model to predict for the sample test data set provided as part of the assignment

results = predict(RFmodel, pml_test_set)
print("The results:")
results
#confusionMatrix(results, pml_test_set$classe)
