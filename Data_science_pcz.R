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

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
#bilot 
pca = prcomp(training_set[,1:7],rank = 2) # perform PCA
screeplot(pca)
biplot(pca)
pca$sdev # standard deviation of the principal components
pca$center # mean of the original variables
pca$scale # standard deviation of the original variables
#interpret the results 

training_set_pca = as.data.frame(predict(pca, training_set))
training_set_pca$Class = training_set$Class
test_set_pca = as.data.frame(predict(pca, test_set))
test_set_pca$Class = test_set$Class

#SVM linear
# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Class ~ .,
                 data = training_set_pca,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set_pca[-3])

# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
accuracy = sum(diag) / n * 100


# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = ' (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = '(Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))