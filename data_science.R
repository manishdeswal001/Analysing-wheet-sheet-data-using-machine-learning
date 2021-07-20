# Data Preprocessing

# Importing the dataset
dataset = read.csv('seeds.csv', fileEncoding="UTF-8-BOM")


# Taking care of missing data
dataset$Area = ifelse(is.na(dataset$Area),
                        ave(dataset$Area, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Area)
dataset$Perimeter = ifelse(is.na(dataset$Perimeter),
                     ave(dataset$Perimeter, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Perimeter)
dataset$Compactness = ifelse(is.na(dataset$Compactness),
                        ave(dataset$Compactness, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Compactness)
dataset$Length.of.kernel = ifelse(is.na(dataset$Length.of.kernel),
                             ave(dataset$Length.of.kernel, FUN = function(x) mean(x, na.rm = TRUE)),
                             dataset$Length.of.kernel)
dataset$Width.of.kernel = ifelse(is.na(dataset$Width.of.kernel),
                                  ave(dataset$Width.of.kernel, FUN = function(x) mean(x, na.rm = TRUE)),
                                  dataset$Width.of.kernel)
dataset$Asymmetry.coefficient = ifelse(is.na(dataset$Asymmetry.coefficient),
                                 ave(dataset$Asymmetry.coefficient, FUN = function(x) mean(x, na.rm = TRUE)),
                                 dataset$Asymmetry.coefficient)
dataset$Length.of.kernel.groov = ifelse(is.na(dataset$Length.of.kernel.groov),
                                       ave(dataset$Length.of.kernel.groov, FUN = function(x) mean(x, na.rm = TRUE)),
                                       dataset$Length.of.kernel.groov)

# Encoding categorical data
dataset$Class = factor(dataset$Class,
                           levels = c('1', '2','3'), # convert categories to numbers
                           labels = c('1', '2','3'))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Class, SplitRatio = 0.8) #dependent variable 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 1:7] = scale(training_set[, 1:7]) #as 2 cloumns are non numberials but factorial
test_set[, 1:7] = scale(test_set[, 1:7])


#Feature Selection top 2

library(mlbench)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Class~., data=dataset, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


#SVM linear
# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

summary(classifier)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-8])

# Making the Confusion Matrix
cm = table(test_set[, 8], y_pred)
confusion_matrix = cm
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
accuracy = sum(diag) / n *100

#visluzaing traning set
plot(classifier,training_set,Area~Perimeter)
#visluzaing test set
plot(classifier,test_set,Area~Perimeter)


# Applying k-Fold Cross Validation for linear kernel SVM
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Class ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'linear')
  y_pred = predict(classifier, newdata = test_fold[-8])
  cm = table(test_fold[, 8], y_pred)
  n = sum(cm) # number of instances
  diag = diag(cm) # number of correctly classified instances per class
  accuracy = sum(diag) / n *100
  return(accuracy)
})
mean_accuracy = mean(as.numeric(cv))

# Applying Grid Search to find the best parameters for Linear KErnel SVM
# install.packages('caret')
library(caret)
classifier = train(form = Class ~ ., data = training_set, method = 'svmLinear')
classifier
classifier$bestTune


#SVM Kernal radial

classifier = svm(formula = Class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')# Predicting the Test set results
summary(classifier)
y_pred = predict(classifier, newdata = test_set[-8])

# Making the Confusion Matrix
cm = table(test_set[, 8], y_pred)
confusion_matrix = cm
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
accuracy = sum(diag) / n *100
#visluzaing traning set
plot(classifier,training_set,Area~Perimeter)
#visluzaing test set
plot(classifier,test_set,Area~Perimeter)

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Class ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-8])
  cm = table(test_fold[, 8], y_pred)
  n = sum(cm) # number of instances
  diag = diag(cm) # number of correctly classified instances per class
  accuracy = sum(diag) / n *100
  return(accuracy)
})
mean_accuracy = mean(as.numeric(cv))


# Fitting Decision Tree Classification to the Training set
# install.packages('rpart')
library(ISLR)
library(tree)
classifier = tree(formula = Class ~ .,
                   data = training_set)

# Plotting the tree #without feature scaling
plot(classifier, training_set,Area~Perimeter)
text(classifier,pretty = 0 )

# check how model is doing
summary(classifier)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-8], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 8], y_pred)
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
accuracy = sum(diag) / n * 100
misclassification_error_rate = mean(y_pred != test_set[, 8])*100


# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = rpart(formula = Class ~ .,
                     data = training_set)
  y_pred = predict(classifier, newdata = test_set[-8], type = 'class')
  cm = table(test_set[, 8], y_pred)
  n = sum(cm) # number of instances
  diag = diag(cm) # number of correctly classified instances per class
  accuracy = sum(diag) / n *100
  return(accuracy)
})
mean_accuracy = mean(as.numeric(cv))

# cross - validation plot
set.seed(3)
cv_Class<-cv.tree(classifier, FUN = prune.misclass)
names(cv_Class)
plot(cv_Class$size, cv_Class$dev, type = "b")

#pruing the tree
pruned_class=prune.misclass(classifier,best=4)
plot(pruned_class)
text(pruned_class,pretty=0)

tree_predict<-predict(pruned_class, test_set, type="class")
new_cm = table(test_set[, 8], tree_predict)
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
new_accuracy = sum(diag) / n * 100
new_misclassification_error_rate = mean(y_pred != test_set[, 8])*100








