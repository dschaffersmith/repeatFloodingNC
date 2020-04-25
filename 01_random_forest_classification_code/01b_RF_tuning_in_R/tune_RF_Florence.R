## Tuning random forest model to map flooding from Hurricane Matthew

## April 8, 2020

## Danica Schaffer-Smith
## d.schaffer-smith@tnc.org


## Load libraries
library(caret)
#library(mlbench)
library(randomForest)
library(rgdal)
library(tidyverse)
library(data.table)
library(mlr)
library(h2o)
library(OOBCurve)

## Set seed to get consistent results
set.seed(45)

## Read in validation dataset
## This dataset can be regenerated as an export from Google Earth Engine (GEE) 
## using the script 'random_forest_Hurricane Florence_earth_engine.txt'.
df <- readOGR('data/validation_set_florence.shp')
df@data$long <- df@coords[, 1]
df@data$lat <- df@coords[, 2]

df <- df@data

## Set up the training and testing partitions to match those in the GEE code
## In the GEE code, the 'random' field was created and populated with random 
## numbers ranging from 0-1 to select a random set of polygon regions.
traindf <- df %>% filter(random < 0.7)
testdf <- df %>% filter(random >=0.7)

## Check for missing values (Not expected as grids in GEE had no missing data)
table(is.na(traindf))
#FALSE 
#57579 
table(is.na(testdf))
#FALSE 
#23647 

## Remove extra variables
traindf <- traindf %>% select(-long, -lat, -random)
testdf <- testdf %>% select(-long, -lat, -random)


## -------------------------------------------------
## Bagging or RF? 
#create a task
traintask <- makeClassifTask(data = traindf,target = "status") 
testtask <- makeClassifTask(data = testdf,target = "status")

#create learner
bag <- makeLearner("classif.rpart",predict.type = "response")
bag.lrn <- makeBaggingWrapper(learner = bag,bw.iters = 100,bw.replace = TRUE)

#set 10 fold cross validation
rdesc <- makeResampleDesc("CV",iters=10L)

# Run a model using bagging
# Test for accuracy
#r <- resample(learner = bag.lrn , task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc) ,show.info = T)
r <- resample(learner = bag.lrn , task = traintask, resampling = rdesc, measures = list(acc) ,show.info = T)
#Aggregated Result: acc.test.mean=0.8695039

## Check the performance of RF w/ 100 trees
#make randomForest learner
rf.lrn <- makeLearner("classif.randomForest")
rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE)
r <- resample(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), show.info = T)
## Aggregated Result: acc.test.mean=0.9093584
## RF performs better than bagging!


## -------------------------------------------------
## We can further adjust parameters with MLR package.
## In GEE we can adjust the number of trees, variables per split (mtry), 
## minimum size of terminal nodes (nodesize), out of bag mode
getParamSet(rf.lrn)

## set parameter space
params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),
                       makeIntegerParam("nodesize",lower = 10,upper = 50), 
                       makeIntegerParam("ntree",lower = 20, upper = 1000))

## set validation strategy
rdesc <- makeResampleDesc("CV",iters=10L)

## set optimization technique
ctrl <- makeTuneControlRandom(maxit = 10L)

## start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)
#[Tune] Result:  mtry=8; nodesize=14; ntree=993 : acc.test.mean=0.9061092


## Run the random forest again with the idealized parameters


## ---------------------------------------------------------
## Identify where no improvement in OOB error / accuracy is gained with increasing trees
# create training and validation data 
#set.seed(123)
#valid_split <- initial_split(ames_train, .8)

# training data
#ames_train_v2 <- analysis(valid_split)

# validation data
#ames_valid <- assessment(valid_split)
#x_test <- ames_valid[setdiff(names(ames_valid), "Sale_Price")]
#y_test <- ames_valid$Sale_Price
x_test <- testdf[setdiff(names(testdf), "status")]
y_test <- testdf$status

rf_oob_comp <- randomForest(
  formula = status ~ .,
  data    = traindf,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  #scale_y_continuous(labels = scales::dollar) +
  #xlim(0,1000) +
  xlab("Number of trees") +
  theme_classic()

# ~ 300 trees looks good given that the error stabalizes there, although more recommended.  
# Test error is lower than OOB error. 


## --------------------------------------------------------------
## GET STATS AND VARIABLE IMPORTANCE FOR FINAL RF MODEL
library(randomForest)

fit <- randomForest(as.factor(status) ~ vvMinBefor + 
    vvMaxAfter +
    vhMinBefor +
    vhMaxAfter + 
    vhVvRatioA + 
    vhVvRatioB + 
    elev + 
    geomorphon + 
    HAND + 
    floodPluvi + 
    floodDefen + 
    percTree +     
    percImperv,
                    data=traindf, 
                    importance=TRUE, 
                    ntree=993, ## Params optimized in previous section
                    mtry=8, 
                    nodesize=14) 

fit
#Call:
#  randomForest(formula = as.factor(status) ~ vvMinBefor + vvMaxAfter +      vhMinBefor + vhMaxAfter + vhVvRatioA + vhVvRatioB + elev +      geomorphon + HAND + floodPluvi + floodDefen + percTree +      percImperv, data = traindf, importance = TRUE, ntree = 993,      mtry = 8, nodesize = 14) 
#Type of random forest: classification
#Number of trees: 993
#No. of variables tried at each split: 8

#OOB estimate of  error rate: 9.15%
#Confusion matrix:
#  1   2   3   4 class.error
#1 839   9  39   4  0.05836139
#2  13 812   2  62  0.08661417
#3  56   0 647  20  0.10511757
#4   2  87  16 779  0.11877828

fit.train <- predict(fit, newdata = traindf)
mean(fit.train == traindf$status) # Overall accuracy for training set: 0.9577797

varImpPlot(fit)

## Assess accuracy on witheld test dataset
fit.test <- predict(fit, newdata = testdf)
table(fit.test, testdf$status)
#fit.test   1   2   3   4
#1 341   5  38   0
#2   6 335   0  34
#3   9   0 258   6
#4   3  21  10 325

mean(fit.test == testdf$status) 
# [1] 0.9051042


## Standardized variable importance plot
importance(fit, type=1, scale=TRUE)
varImpPlot(fit, type = 1, scale = T)

