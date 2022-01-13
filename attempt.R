library(data.table)
library(ggplot2)
library(lattice)
library(caret)
library(Matrix)
library(glmnet)
library(mlbench)
library(MLmetrics)
library(pROC)

##load file and set format
f1M<-read.csv("factor1M.csv") ##binary outcome: 1 for copd, 0 for ipf
f1M
f1m<-as.data.frame(f1M)
rownames(f1m)<-f1M[,1]
f1m<-f1m[,-1]
f1m
##split the data
training.samples<-createDataPartition(f1m$Disease,p=0.6,list=F)
train<-f1m[training.samples,]
test<-f1m[-training.samples,]
##creating dummy variable
x<-model.matrix(Disease~.,train)[,-1]
##setting oucome to numeric
y<-ifelse(train$Disease=="copd",1,0)
##training the model
lambdas = 10^seq(-3, -1, length = 20)
trControl = trainControl(
  method = 'repeatedcv', 
  number = 10, 
  repeats = 5, 
  search = 'grid'
)
tuneGrid = expand.grid(alpha = 1, lambda = lambdas)
lasso = train(
  Disease ~ ., data = train, 
  method = 'glmnet',
  trControl = trControl, 
  tuneGrid = tuneGrid
)
lasso

#Tuning parameter 'alpha' was held constant at a value of 1
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were alpha = 1 and lambda = 0.06158482.

##measure performance of training data
p<-predict(lasso,train)
error<-p-train$Disease
sqrt(mean(error^2))
# 0.4298735

##measure performance of testing data
p<-predict(lasso,test)
error<-p-test$Disease
sqrt(mean(error^2))
# 0.4956424

##measure performance for the whole dataset
p<-predict(lasso,f1m)
error<-p-f1m$Disease
sqrt(mean(error^2))
#0.4573175

#creating dummy categorical variables
x <- model.matrix(Disease~., train)[,-1]
##Using the value of lambda from our lasso model, retraining 
model<-glmnet(x,train$Disease,alpha=1,family="binomial",lambda=0.06158482)
coef(model)
#9 x 1 sparse Matrix of class "dgCMatrix"
#s0
#(Intercept)     -0.98478095
#Cer.d18.1.23.0.  .         
#Cer.d18.1.24.0.  .         
#Cer.d18.1.22.0.  .         
#PC.O.38.6.       .         
#PC.40.4.         .         
#PC.40.5.        -0.5694956
#PC.O.36.5.       0.4423868
#SM.d18.1.14.0.   0.3893210

x.test<-model.matrix(Disease~.,test)[,-1]
probabilities<-predict(model,newx = x.test)
predicted.classes<-ifelse(probabilities>0.5,1,0)
observed.classes<-test$Disease
mean(predicted.classes==observed.classes)
#0.55
#find the best lambda using cross validation
cv.lasso<-cv.glmnet(x,train$Disease,alpha=1,family="binomial")
#fit the final model on the training data
model<-glmnet(x,train$Disease,alpha=1,amily="binomial",lambda=cv.lasso$lambda.min)
coef(model)
#9 x 1 sparse Matrix of class "dgCMatrix"
#s0
#(Intercept)      0.25364795
#Cer.d18.1.23.0.  .         
#Cer.d18.1.24.0.  .         
#Cer.d18.1.22.0.  .         
#PC.O.38.6.       .         
#PC.40.4.         .         
#PC.40.5.        -0.10635044
#PC.O.36.5.       0.07798797
#SM.d18.1.14.0.   0.07540385
# Make predictions on the test data
x.test <- model.matrix(Disease ~., test)[,-1]
probabilities <- model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- test$Disease
mean(predicted.classes == observed.classes)
#0.6
# Final model with lambda.min
cv.lasso$lambda.min
#0.07324422
lasso.model <- glmnet(x, train$Disease, alpha = 1, family = "binomial",lambda = 0.07324422)
# Make prediction on test data
x.test <- model.matrix(Disease ~., test)[,-1]
probabilities <- lasso.model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)
# Model accuracy
observed.classes <- test$Disease
mean(predicted.classes == observed.classes)
#0.5
#recompute coefs
#coef(cv.lasso, cv.lasso$lambda.min)
#9 x 1 sparse Matrix of class "dgCMatrix"
#s1
#(Intercept)     -1.1793858
#Cer.d18.1.23.0.  .        
#Cer.d18.1.24.0.  .        
#Cer.d18.1.22.0.  .        
#PC.O.38.6.       .        
#PC.40.4.         .        
#PC.40.5.        -0.4155385
#PC.O.36.5.       0.3905689
#SM.d18.1.14.0.   0.3237240

##Using glm for AUC since no other method works

model<-glm(Disease~.,family="binomial",train)
p<-predict(model,test,type="response")
p
c_or_i<-ifelse(p>0.5,1,0)
p_class<-factor(c_or_i,levels=levels(test$Disease))
confusionMatrix(p_class,test$Disease)
confusionMatrix(p_class, test[["Disease"]])
library(caTools)
colAUC(p,test[["Disease"]],plotROC = T)
#copd vs. ipf 0.75