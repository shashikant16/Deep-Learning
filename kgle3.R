library(data.table)
library(knitr)
library(tidyverse)
library(stringr)
library(car)
library(anchors)
library(GGally)
library(h2o)
library(dplyr)
setwd("C:/Users/Administrator/Desktop/Kaggle/Deep_lerning")
train <- fread("train.csv",colClasses = "numeric",verbose = FALSE)
test <- fread("./test.csv",colClasses = "numeric",verbose=FALSE)

train <- train %>% dplyr::select(-id)
# Use the base summary function for result summaries not dplyr. This will provide us with the ranges of the variables including the minimums
s <- summary(train)  
#  Extract and view the min values that have -1 from the summary we just created use str_detect from stringr package. 
s %>% 
  data.frame() %>% 
  filter(str_detect(Freq,"-1")) %>% 
  filter(str_detect(Freq,"Min")) %>%
  dplyr::select(-1)
# Replace the -1 ie missing values in the numeric columns with the mean of the respective column using recode, first replacing the -1 with NAs otherwise the -1 will distort the calculated mean
# Find the index of the columns names that contain cat
indx <- grepl('cat', colnames(train))
# First convert the categorical columnds to NA
train <- replace.value( train, colnames(train)[indx], from=-1, to=NA, verbose = FALSE)
# Sanity check
round(mean(train$ps_car_03_cat,na.rm = TRUE)) #-1,-1,-1,0,-1

round(mean(train$ps_car_05_cat,na.rm = TRUE)) #1,-1,-1,1

# Create the columns mean functions for catageorical means
roundmean <- function(x) {replace(x, is.na(x), round(median(x, na.rm = TRUE))) } 
# Replace the NAs with the roundmean
train <- as.data.frame(apply(train, 2, roundmean))
#Sanity check
train$ps_car_03_cat[2]

train$ps_car_05_cat[2]
train <- replace.value( train, colnames(train)[-indx], from=-1, to=NA, verbose = FALSE)
#Sanity check this is same mean as calculated above before recoding
median(train$ps_reg_03,na.rm = TRUE) # row 3
# Create the columns mean functions , one for mean of the continuous numerical columns
justmean <- function(x) {replace(x, is.na(x), mean(x, na.rm = TRUE)) }
# Replace the NAs with the justmean
train <- as.data.frame(apply(train, 2, justmean))
#Sanity check this is same mean as calculated above before recoding
train$ps_reg_03[3]

# Sanity check that we have cleaned up all -1's, the result should be empty
colsum <- colSums(train=="-1") 
colsum[colsum>0]

s_test <- summary(test)  
#  Extract and view the min values that have -1 from the summary we just created use str_detect from stringr package. 
s_test %>% 
  data.frame() %>% 
  filter(str_detect(Freq,"-1")) %>% 
  filter(str_detect(Freq,"Min")) %>%
  dplyr::select(-1)

train$target <- as.factor(train$target)
# Set seed for reproduceability
set.seed(123)
h2o.init(port = 54321,nthreads = -1)

# Transfer data to h2o using the as.h2o function
train.hex = as.h2o(train,  destination_frame ="train")

# Create a y variable with the outcome or dependent target
y = "target"
# We have already removed the id variable so the remaining variables will be the independent variables
x = colnames(train.hex[,-1])

# Glm logistic model using h2o
set.seed(123) # to ensure results are reproducable
system.time(glm <- h2o.glm(x=x, 
                           y=y, 
                           training_frame=train.hex,
                           nfolds=5,# Defaults to 0
                           keep_cross_validation_predictions=TRUE, # Defaults to FALSE
                           fold_assignment = "Stratified", # Defaults to AUTO
                           family="binomial" # Defaults to gaussian.
)
)
# Let's take a look at the results of the glm model
h2o.performance(glm)

h2o.varimp(glm)

set.seed(123) # to ensure results are reproducable
# Create a randomforest model using h2o
system.time(forest <- h2o.randomForest(x=x, 
                                       y=y, 
                                       training_frame=train.hex,
                                       nfolds = 5, # Defaults to 0 which disables the CV
                                       max_depth=10, # Defaults to 20
                                       ntrees=25, # Defaults to 50
                                       keep_cross_validation_predictions=TRUE, # Defaults to FALSE
                                       fold_assignment="Stratified", # The 'Stratified' option will stratify the folds based on the response variable, for classification problems Defaults to AUTO
                                       seed = 123)
)
# Let's take a look at the results of the gbm model
h2o.performance(forest)
h2o.varimp(forest)
#plot(forest,timestep="number_of_trees",metric="RMSE")
#plot(forest,timestep="number_of_trees",metric="AUC")

set.seed(123) # to ensure results are reproducable
# Train and cross validate a gbm model using h2o
system.time(gbm <- h2o.gbm(x=x, 
                           y=y, 
                           training_frame=train.hex,
                           nfolds = 5,# Defaults to 0 which disables the CV
                           distribution = "bernoulli",
                           ntrees = 100, # Defaults to 50 
                           max_depth = 5, # Defaults to 5
                           min_rows = 10, # Deaults to 10
                           learn_rate = 0.01, # Defaults to 0.1
                           keep_cross_validation_predictions=TRUE, # Defaults to FALSE
                           fold_assignment="Stratified", # The 'Stratified' option will stratify the folds based on the response variable, for classification problems. Defaults to AUTO
                           seed = 123)
)
# Let's take a look at the results of the gbm model
h2o.performance(gbm)
h2o.varimp(gbm)
set.seed(123) # to ensure results are reproducable
system.time(deep <- h2o.deeplearning(x = x,  # column numbers for predictors
                                     y = y,   # column name for label
                                     training_frame = train.hex, # data in H2O format
                                     nfolds = 5, # Defaults to 0 which disables the CV
                                     fold_assignment = "Stratified",# The 'Stratified' option will stratify the folds based on the response variable, for classification problems. Defaults to AUTO
                                     activation = "Rectifier" ) # the activation function. Defaults to Rectifier.
)
h2o.performance(deep)
h2o.varimp(deep)
#plot(deep,timestep="epochs",metric="RMSE")
#plot(deep,timestep="epochs",metric="AUC")
# library(xgboost)
# system.time(xgboost <- h2o.xgboost(x=x,
#              y=y,
#              training_frame=train.hex,
#              nfolds = 5,# Defaults to 0 which disables the CV
#              distribution = "bernoulli",
#              ntrees = 100, # Defaults to 50
#              max_depth = 5, # Defaults to 5
#              min_rows = 10, # Deaults to 10
#              learn_rate = 0.01, # Defaults to 0.1
#              keep_cross_validation_predictions=TRUE, # Defaults to FALSE
#              fold_assignment="Stratified", # The 'Stratified' option will stratify the folds based on the response variable, for classification problems Defaults to AUTO
#              seed = 123)
#              )
# #Let's take a look at the results of the gbm model
# h2o.performance(xgboost)
# h2o.varimp(xgboost)
basemodels <- list(glm, gbm,forest)
system.time(ensemble <- h2o.stackedEnsemble(x = x, 
                                            y = y, 
                                            training_frame = train.hex,
                                            base_models = basemodels)
)
h2o.performance(ensemble)


# Plot the model RMSE
# rmse_models<- c(h2o.rmse(glm),h2o.rmse(forest),h2o.rmse(gbm),h2o.rmse(deep),NA,h2o.rmse(ensemble))
# names(rmse_models)<- c("glm","forest","gbm","deep","xgboost","ensemble")
# barplot(sort(rmse_models,decreasing = TRUE),main = "Comparison of Model RMSE")
# # Plot the model AUCs
# auc_models<- c(h2o.auc(glm),h2o.auc(forest),h2o.auc(gbm),h2o.auc(deep),NA,h2o.auc(ensemble))
# names(auc_models)<- c("glm","forest","gbm","deep","xgboost","ensemble")
# barplot(sort(auc_models,decreasing = TRUE),main = "Comparison of Model AUCs")
# #Plot the model Ginis
# gini_models<- c(h2o.giniCoef(glm),h2o.giniCoef(forest),h2o.giniCoef(gbm),h2o.giniCoef(deep),NA,h2o.giniCoef(ensemble))
# names(gini_models)<- c("glm","forest","gbm","deep","xgboost","ensemble")
# barplot(sort(auc_models,decreasing = TRUE),main = "Comparison of Model Gini Coefficients")
# # Plot system time
# systime_models<- c(75.6,456.48,1992.67,5076.37,0,25.39)
# names(systime_models)<- c("glm","forest","gbm","deep","xgboost","ensemble")
# barplot(sort(systime_models),main = "Comparison of Model Elapsed Time")

test.hex = as.h2o(test)
# Make predictions 
preds = as.data.frame(h2o.predict(ensemble, test.hex))
# Create Kaggle Submission File
my_solution <- data.frame(id = test$id, target = preds$predict)
my_solution$id <- as.integer(my_solution$id)
# Write solution to file portoglmh20.csv/'
fwrite(my_solution, "portoEnsembleh20_3.csv", row.names = F)

