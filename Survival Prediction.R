
## First Run Only
## install.packages("caret", dependencies = TRUE)
## install.packages("randomForest")
rm(list=ls())
library(caret)
library(randomForest)
## Set Wotrking Directory

setwd("D:\\Users\\KAMI\\Downloads")

## Reading Input Files
Trainingset <- read.table("train.csv", sep = ",", header = TRUE)
Testset <- read.table("test.csv", sep = ",", header = TRUE)



## Checking for a relation Between Survival and Class of Travel
table(Trainingset[,c("Survived", "Pclass")])

## Checking few more variables for Relation against Survival.
library(fields)
bplot.xy(Trainingset$Survived, Trainingset$Age)
## will ignore due to no clear relation.

bplot.xy(Trainingset$Survived, Trainingset$Fare)
# Will use due to Clear relation between survival and fare.
summary(Trainingset$Fare)

# Convert Survived to type Factor
Trainingset$Survived <- factor(Trainingset$Survived)

# Train the model using Cart , KNN , SVM , Rf Algos
library(e1071)

# CART
set.seed(7)
model.cart <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare, data = Trainingset, method="rpart",  trControl = trainControl(method = "cv", number = 10))
# kNN
set.seed(7)
model.knn <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare, data = Trainingset, method="knn",  trControl = trainControl(method = "cv", number = 10))
# SVM
set.seed(7)
model.svm <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare, data = Trainingset, method="svmRadial",  trControl = trainControl(method = "cv", number = 10))
# Random Forest
set.seed(7)
model.rf <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare, data = Trainingset, method="rf",  trControl = trainControl(method = "cv", number = 10))

Multiresult <- resamples(list(cart=model.cart, knn=model.knn, svm=model.svm, rf=model.rf))
summary(Multiresult)



#### FEATURE ENGINEERING 
## Combining the Datasets to create uniform Factor levels throught data.
Testset$Survived=NA
Combined <- rbind(Trainingset, Testset)
Combined$Name <- as.character(Combined$Name)
## Exctracting Titles of each Memeber
Combined$Title <- sapply(Combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

Combined$Title <- sub(' ', '', Combined$Title)

## Normalizing Titles
Combined$Title[Combined$Title %in% c('Mme', 'Mlle','Ms')] <- 'Ms'

Combined$Title[Combined$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Master','Dr','Col')] <- 'Sir'

Combined$Title[Combined$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

Combined$Title <- factor(Combined$Title)

## Creating a New variable by adding Sibling+ Parents + self

Combined$FamilySize <- Combined$SibSp + Combined$Parch + 1
## obtaining Surnames to concatenate to above variable.
Combined$Surname <- sapply(Combined$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

Combined$FamilyID <- paste(as.character(Combined$FamilySize), Combined$Surname, sep="")

## Creating Normalized Levels for FamilyID variable.
Combined$FamilyID[Combined$FamilySize <= 2] <- 'Small'
## Handling Special cases which are missed .
FamilyIDs <- data.frame(table(Combined$FamilyID))

FamilyIDs <- FamilyIDs[FamilyIDs$Freq <= 2,]
Combined$FamilyID[Combined$FamilyID %in% FamilyIDs$Var1] <- 'Small'
Combined$FamilyID <- factor(Combined$FamilyID)
## Splitting data back into original data sets.
train <- Combined[1:nrow(Trainingset),]
test <- Combined[(nrow(Trainingset)+1):nrow(Combined),]

## Rerunning the Algorithims 

set.seed(7)
model.cart <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare+Title + FamilySize + FamilyID, data = train, method="rpart",  trControl = trainControl(method = "cv", number = 50))
# kNN
set.seed(7)
model.knn <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare+Title + FamilySize + FamilyID, data = train, method="knn",  trControl = trainControl(method = "cv", number = 50))
# SVM
set.seed(7)
model.svm <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare+Title + FamilySize + FamilyID, data = train, method="svmRadial",  trControl = trainControl(method = "cv", number = 50))
# Random Forest
set.seed(7)
model.rf <- train(Survived ~ Pclass + Sex + SibSp +Embarked + Parch + Fare+Title + FamilySize + FamilyID, data = train, method="rf",  trControl = trainControl(method = "cv", number = 50))

Multiresult <- resamples(list( cart=model.cart, knn=model.knn, svm=model.svm, rf=model.rf))
summary(Multiresult)


## Predicting Based on RF
test$Survived <- predict(model.rf, newdata = test)

## Summary of Model
summary(test)

## Handling Na's
test$Fare <- ifelse(is.na(test$Fare), mean(test$Fare, na.rm = TRUE), test$Fare)

## Rerunning Prediction based on RF
test$Survived <- predict(model.rf, newdata = test)
## Writing the Prediction Output file
write.table(test, file = "Pred_result.csv", col.names = TRUE, row.names = FALSE, sep = ",")
## Creating Submission Result to upload on Kaggle. Achieved a 80% accuracy. (Need to work on reaching 90%)
submission <- test[,c("PassengerId", "Survived")]
write.table(submission, file = "submission.csv", col.names = TRUE, row.names = FALSE, sep = ",")
