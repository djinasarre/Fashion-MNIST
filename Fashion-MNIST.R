#We study a Kaggle.com file is the Zalando fashion mnist 

#fashion_mnist_test https://www.kaggle.com/zalando-research/fashionmnist/download/w9pAJQIODXpHyv0Dccb2%2Fversions%2Fe9j8R2sqo2KjiyQl77np%2Ffiles%2Ffashion-mnist_test.csv?datasetVersionNumber=1
#fashion_mnist_test https://www.kaggle.com/zalando-research/fashionmnist/download/TZyWaiHnZ7ZtpiUn3ejX%2Fversions%2F6az5Tsr9ELxEsxhUEWrp%2Ffiles%2Ffashion-mnist_train.csv?datasetVersionNumber=1

#Context
#Fashion-MNIST is a dataset of Zalando's article images-consisting of a training set of 60,000 
#examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with
# a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for
# the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image 
#size and structure of training and testing splits.
#Acknowledgements
#Original dataset was downloaded from https://github.com/zalandoresearch/fashion-mnist

#Dataset was converted to CSV with this script: https://pjreddie.com/projects/mnist-in-csv/

#Labels
#Each training and test example is assigned to one of the following labels:
#0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag
#9 Ankle boot
library(readr)
fashion_mnist_test <- read_csv("Data Science/Capstone proyect/fashion-mnist_test.csv")
View(fashion_mnist_test)
library(readr)
fashion_mnist_train <- read_csv("Data Science/Capstone proyect/fashion-mnist_train.csv")
View(fashion_mnist_train)

class(fashion_mnist_train)

# We change vectors fashion_mnist_train_images and fashion_mnist_test_images classes as matrix:
# Image Data is transformed to matrix because they are easier indexable.
fashion_mnist_train_images <- as.matrix(fashion_mnist_train[, 2:dim(fashion_mnist_train)[2]])
fashion_mnist_test_images <- as.matrix(fashion_mnist_test[, 2:dim(fashion_mnist_test)[2]]) 
view(fashion_mnist_test_images)
library(tidyverse)
library(dslabs)
#The dataset includes two components, a train set (fashion_mnist_train) and a test set (fashion_mnist_test):
#Each of these components includes now a matrix (60000 x 784)with features in the columns:
dim(fashion_mnist_train_images)
class(fashion_mnist_train_images)


#and change vectors fashion_mnist_train$label and fashion_mnist_test$label classes as integers:
class(fashion_mnist_train$label)
fashion_mnist_train$label <- type.convert(fashion_mnist_train$label)
class(fashion_mnist_train$label)
class(fashion_mnist_test$label)
fashion_mnist_test$label <- type.convert(fashion_mnist_test$label)
class(fashion_mnist_test$label)
table(fashion_mnist_train$label)
#We consider a subset of the dataset. We will sample 10,000 random rows from
# the training set and 1,000 random rows from the test set, to run in less time:
set.seed(1990, sample.kind="Rounding")
index <- sample(nrow(fashion_mnist_train_images), 10000)
x <- fashion_mnist_train_images[index,]
y <- factor(fashion_mnist_train$label[index])

index <- sample(nrow(fashion_mnist_test_images), 1000)
x_test <- fashion_mnist_test_images[index,]
y_test <- factor(fashion_mnist_test$label[index])
#We run the nearZero function from the caret package to see that several 
#features do not vary much from observation to observation. We can see that there
# are a large number of features with 0 variability:
library(matrixStats)
sds <- colSds(x)
qplot(sds, bins = 256)
#Parts of the image rarely contain writing
# (dark pixels).
#We apply nzv caret package function to recommends features to be removed 
#due to near zero variance:
library(caret)
nzv <- nearZeroVar(x)
#We can see the columns recommended for removal:
image(matrix(1:784 %in% nzv, 28,28))
#So we end up keeping this number of columns:
col_index <- setdiff(1:ncol(x), nzv)
length(col_index)
#Now we are ready to fit some models. Before we start, we need to add column names to
# the feature matrices as these are required by caret:

colnames(x) <- 1:ncol(fashion_mnist_train_images)
colnames(x_test) <- colnames(fashion_mnist_train_images)
#We will start with kNN. The first step is to optimize for  k
#To compute a distance between each observation in the test set and each observation in the 
#training set we will therefore use k-fold cross validation to improve speed.
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[,col_index], y, 
method = "knn", 
tuneGrid = data.frame(k = c(3,5,7)),
trControl = control)
train_knn
#Now that we optimize our algorithm, we can fit it to the entire dataset:
fit_knn <- knn3(x[, col_index], y,  k = 3)
y_hat_knn <- predict(fit_knn, x_test[, col_index], type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]

#We now achieve an accuracy of about 0.758. From the specificity and sensitivity, we
# also see that Class 6 are the hardest to detect and the most commonly incorrectly 
#predicted.
cm$byClass[,1:2]
# Now we try to increase accuracy with random forest algorithm.
library(Rborist)
control <- trainControl(method="cv", number = 5, p = 0.8)
grid <- expand.grid(minNode = c(1) , predFixed = c(10, 15, 35))

train_rf <-  train(x[, col_index], 
                   y, 
                   method = "Rborist", 
                   nTree = 50,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)

ggplot(train_rf)
train_rf$bestTune
#we are now going to fit our final model:
fit_rf <- Rborist(x[, col_index], y, 
                  nTree = 1000,
                  minNode = train_rf$bestTune$minNode,
                  predFixed = train_rf$bestTune$predFixed)
y_hat_rf <- factor(levels(y)[predict(fit_rf, x_test[ ,col_index])$yPred])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]
#We  get our higher accuracy 0,823
#Finally I want to computes the importance of each feature:
library(randomForest)
rf <- randomForest(x, y,  ntree = 50)
imp <- importance(rf)
#We can see which features are most being used by plotting an image:
image(matrix(imp, 28, 28))
imp

p_rf <- predict(fit_rf, x_test[,col_index])$census  
p_rf<- p_rf / rowSums(p_rf)
p_knn  <- predict(fit_knn, x_test[,col_index])
p <- (p_rf + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)
confusionMatrix(y_pred, y_test)$overall["Accuracy"]
#Computing new class probabilities by taking the average of random forest and kNN. 
#You can see that the accuracy improves to 0.793.

#Finally when we fit our final model with random forest algorithm we  get our higher accuracy 0,823.

##Fashion Mnist dimension reduction
#We expect pixels close to each other on the grid to be correlated, so dimension reduction is possible.
#We try PCA and explore the variance of the PCs.
col_means <- colMeans(fashion_mnist_test_images)
pca <- prcomp(fashion_mnist_train_images)
pc <- 1:ncol(fashion_mnist_test_images)
qplot(pc, pca$sdev)
#First few PCs explain a large percent of the variability.
summary(pca)$importance[,1:5] 
# The first two PCs we see information about the class. Here is a random sample of 3,000 digits:
data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2],
           label=factor(fashion_mnist_train$label)) %>%
  sample_n(3000) %>% 
  ggplot(aes(PC1, PC2, fill=label))+
  geom_point(cex=3, pch=21)
#We see the linear combinations on the grid to get an idea of what is getting weighted:
library(RColorBrewer)
tmp <- lapply( c(1:4,781:784), function(i){
  expand.grid(Row=1:28, Column=1:28) %>%
    mutate(id=i, label=paste0("PC",i), 
           value = pca$rotation[,i])
})
tmp <- Reduce(rbind, tmp)

tmp %>% filter(id<5) %>%
  ggplot(aes(Row, Column, fill=value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradientn(colors = brewer.pal(9, "RdBu")) +
  facet_wrap(~label, nrow = 1)
#The lower variance PCs appear related to unimportant variability in the corners:
tmp %>% filter(id>5) %>%
  ggplot(aes(Row, Column, fill=value)) +
  geom_raster() +
  scale_y_reverse() +
  scale_fill_gradientn(colors = brewer.pal(9, "RdBu")) +
  facet_wrap(~label, nrow = 1)
#We try 36 dimensions since this is about 84% of Cumulative Proportion of the data
summary(pca)$importance[,1:36] 
#First fit the model:
library(caret)
k <- 36
x_train <- pca$x[,1:k]
y <- factor(fashion_mnist_train$label)
fit <- knn3(x_train, y)
#Now transform the test set:
x_test <- sweep(fashion_mnist_test_images, 2, col_means) %*% pca$rotation
x_test <- x_test[,1:k]
#And predict and see how we do:
y_hat <- predict(fit, x_test, type = "class")
confusionMatrix(y_hat, factor(fashion_mnist_test$label))$overall["Accuracy"]
# The accuracy achieved with reduced dimensions almost 0.86



###We want to see how to differentiate Class 6 shirt which is hardest to detect and the most commonly incorrectly
#predicted with 0 T-shirt / top.

set.seed(1991, sample.kind="Rounding")
index_06 <- sample(which(fashion_mnist_train$label %in% c(0,6)), 2000)
y <- fashion_mnist_train$label[index_06] 
x <- fashion_mnist_train_images[index_06,]
index_train <- createDataPartition(y, p=0.8, list = FALSE)

# get the quadrants
# temporary object to help figure out the quadrants
row_column <- expand.grid(row=1:28, col=1:28) 
upper_left_ind <- which(row_column$col <= 14 & row_column$row <= 14)
lower_right_ind <- which(row_column$col > 14 & row_column$row > 14)

# binarize the values. Above 200 is ink, below is no ink
x <- x > 200 

# cbind proportion of pixels in upper right quadrant and
## proportion of pixels in lower right quadrant
x <- cbind(rowSums(x[ ,upper_left_ind])/rowSums(x), 
           rowSums(x[ ,lower_right_ind])/rowSums(x)) 

train_set <- data.frame(y = factor(y[index_train]),
                        x_1 = x[index_train,1],
                        x_2 = x[index_train,2])
test_set <- data.frame(y = factor(y[-index_train]),
                       x_1 = x[-index_train,1],
                       x_2 = x[-index_train,2])

#We use the training set to build a model with several of the models 
#available from the caret package. We will test out all of the 
#following models

models <- c("glm", "lda",  "naive_bayes",  "svmLinear", 
            "gamboost",  "gamLoess", "qda", 
            "knn", "kknn", "loclda", "gam",
            "rf", "ranger",  "wsrf", "Rborist", 
            "avNNet", "mlp", "monmlp",
            "adaboost", "gbm",
            "svmRadial", "svmRadialCost", "svmRadialSigma")
fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = train_set)
}) 

# We use sapply or to create a matrix (rows:400 colums:23) of predictions for the test set
pred <- sapply(fits, function(object) 
  predict(object, newdata = test_set))
dim(pred)
# Compute accuracy for each model on the test set and look for the 
#mean accuracy across all models.
acc <- colMeans(pred == test_set$y)
acc
which.max(acc)
mean(acc)
#We see that our maximum accuracy is achieved with the gam model with 0.6350. Being the 
#average of the models 0.5836957.
