cat("\f")
rm(list = ls())
graphics.off()

library(tidyverse)
library(randomForest)
library(glmnet)
library(bayestestR)

# Read and clean up the dataset

spambase <- read_csv("~/Downloads/spambase/spambase.data", 
                     col_names = FALSE)
name = read_table("spambase.names", col_names = TRUE, skip_empty_rows = TRUE)
name1 = name[c(30:86),]
colnames(name1) = "X"
name1 = name1 %>% separate(X, c("var","type"), sep = ":")
class = c("class","categorical")
name1 = rbind(name1,class)
colnames(spambase) = name1$var
spambase %>% group_by(class) %>% count
sum(is.na(spambase))

n       = dim(spambase)[1]
p       = dim(spambase)[2]

X       = data.matrix(spambase[,-p])
y       = as.factor(spambase$class)

n.train = floor(0.9*n)
n.test  = n - n.train
p.train = p - 1

#loop

M               = 50
auc.train.rd    = rep(0,M)
auc.test.rd     = rep(0,M)
auc.train.ls    = rep(0,M)
auc.test.ls     = rep(0,M)
auc.train.en    = rep(0,M)
auc.test.en     = rep(0,M)
auc.train.rf    = rep(0,M)
auc.test.rf     = rep(0,M)

thrs            = seq(0,1,by = 0.01)
I               = length(thrs)


for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  P.train          =     sum(y.train == 1)
  N.train          =     sum(y.train == 0)
  P.test           =     sum(y.test == 1)
  N.test           =     sum(y.test == 0)
  
  #RIDGE
start_time_rd_cv       =     Sys.time()
  ridge.cv             =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 0,  intercept = TRUE,   nfolds = 10, type.measure="auc")
end_time_rd_cv         =     Sys.time()
cv_time_rd             =     end_time_rd_cv - start_time_rd_cv
  ridge                =     glmnet(X.train, y.train, lambda = ridge.cv$lambda[which.max(ridge.cv$cvm)], family = "binomial", alpha = 0,  intercept = TRUE)
  p.hat.train          =     predict(ridge, newx = X.train, type = "response")
  p.hat.test           =     predict(ridge, newx = X.test, type = "response")
 
        # Nested loop for a sequence of threshold
  FPR.train.rd         = rep(0,I)
  TPR.train.rd         = rep(0,I)
  FPR.test.rd          = rep(0,I)
  TPR.test.rd          = rep(0,I)
  
  for (i in c(1:I)){
    th                      =     thrs[i]
    y.hat.train             =     ifelse(p.hat.train > th, 1, 0)
    FP.train                =     sum(y.hat.train[y.train == 0] == 1)
    TP.train                =     sum(y.train[y.hat.train == 1] == 1)
    FPR.train.rd[i]         =     FP.train/N.train
    TPR.train.rd[i]         =     TP.train/P.train
    
    y.hat.test              =     ifelse(p.hat.test > th, 1, 0)
    FP.test                 =     sum(y.hat.test[y.test == 0] == 1)
    TP.test                 =     sum(y.test[y.hat.test == 1] == 1)
    FPR.test.rd[i]          =     FP.test/N.test
    TPR.test.rd[i]          =     TP.test/P.test
  }
  auc.train.rd[m]         =     area_under_curve(FPR.train.rd,TPR.train.rd, method = "trapezoid")
  auc.test.rd[m]          =     area_under_curve(FPR.test.rd,TPR.test.rd, method = "trapezoid") 
  cat(sprintf("m=%.0f|Ridge: auc.train= %.3f| auc.test= %.3f \n\n", 
              m,auc.train.rd[m],auc.test.rd[m]))

  #LASSO
start_time_ls_cv       =     Sys.time()
  lasso.cv             =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 1,  intercept = TRUE,  nfolds = 10, type.measure="auc")
end_time_ls_cv         =     Sys.time()  
cv_time_ls             =     end_time_ls_cv - start_time_ls_cv
  lasso                =     glmnet(X.train, y.train, lambda = lasso.cv$lambda[which.max(lasso.cv$cvm)], family = "binomial", alpha = 1,  intercept = TRUE)
  p.hat.train          =     predict(lasso, newx = X.train, type = "response")
  p.hat.test           =     predict(lasso, newx = X.test, type = "response")
  
           # Nested loop for a sequence of thresholds
  FPR.train.ls  = rep(0,I)
  TPR.train.ls  = rep(0,I)
  FPR.test.ls   = rep(0,I)
  TPR.test.ls   = rep(0,I)
  
  for (i in c(1:I)){
    th                      =     thrs[i]
    y.hat.train             =     ifelse(p.hat.train > th, 1, 0)
    FP.train                =     sum(y.hat.train[y.train == 0] == 1)
    TP.train                =     sum(y.train[y.hat.train == 1] == 1)
    FPR.train.ls[i]         =     FP.train/N.train
    TPR.train.ls[i]         =     TP.train/P.train
    
    y.hat.test              =     ifelse(p.hat.test > th, 1, 0)
    FP.test                 =     sum(y.hat.test[y.test == 0] == 1)
    TP.test                 =     sum(y.test[y.hat.test == 1] == 1)
    FPR.test.ls[i]          =     FP.test/N.test
    TPR.test.ls[i]          =     TP.test/P.test
  }
  auc.train.ls[m]         =     area_under_curve(FPR.train.ls,TPR.train.ls, method = "trapezoid")
  auc.test.ls[m]          =     area_under_curve(FPR.test.ls,TPR.test.ls, method = "trapezoid") 
  cat(sprintf("m=%.0f|Lasso: auc.train= %.3f| auc.test= %.3f \n\n", 
              m,auc.train.ls[m],auc.test.ls[m]))

  
  # ELASTIC NET
start_time_en_cv       =     Sys.time()
  elnet.cv             =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 0.5,  intercept = TRUE,  nfolds = 10, type.measure="auc")
end_time_en_cv         =     Sys.time() 
cv_time_en             =     end_time_en_cv - start_time_en_cv
  elnet                =     glmnet(X.train, y.train, lambda = elnet.cv$lambda[which.max(elnet.cv$cvm)], family = "binomial", alpha = 0.5,  intercept = TRUE)
  p.hat.train          =     predict(elnet, newx = X.train, type = "response")
  p.hat.test           =     predict(elnet, newx = X.test, type = "response")
  
               # Nested loop for a sequence of thresholds
  FPR.train.en   = rep(0,I)
  TPR.train.en   = rep(0,I)
  FPR.test.en    = rep(0,I)
  TPR.test.en    = rep(0,I)
  
  for (i in c(1:I)){
    th                      =     thrs[i]
    y.hat.train             =     ifelse(p.hat.train > th, 1, 0)
    FP.train                =     sum(y.hat.train[y.train == 0] == 1)
    TP.train                =     sum(y.train[y.hat.train == 1] == 1)
    FPR.train.en[i]         =     FP.train/N.train
    TPR.train.en[i]         =     TP.train/P.train
    
    y.hat.test              =     ifelse(p.hat.test > th, 1, 0)
    FP.test                 =     sum(y.hat.test[y.test == 0] == 1)
    TP.test                 =     sum(y.test[y.hat.test == 1] == 1)
    FPR.test.en[i]          =     FP.test/N.test
    TPR.test.en[i]          =     TP.test/P.test
  }
  auc.train.en[m]         =     area_under_curve(FPR.train.en,TPR.train.en, method = "trapezoid")
  auc.test.en[m]          =     area_under_curve(FPR.test.en,TPR.test.en, method = "trapezoid") 
  cat(sprintf("m=%.0f|Elastic Net: auc.train= %.3f| auc.test= %.3f \n\n", 
              m,auc.train.en[m],auc.test.en[m]))
  
  # RANDOM FOREST
  rf                   =     randomForest(X.train,y.train, mtry = sqrt(p.train), importance = TRUE)
  p.hat.train          =     predict(rf, X.train, type = "prob")[,2]
  p.hat.test           =     predict(rf, X.test, type = "prob")[,2]
  
           #Nested Loop for a sequence of threshold
  FPR.train.rf   = rep(0,I)
  TPR.train.rf   = rep(0,I)
  FPR.test.rf    = rep(0,I)
  TPR.test.rf    = rep(0,I)
  
  for (i in c(1:I)){
    th                      =     thrs[i]
    y.hat.train             =     ifelse(p.hat.train > th, 1, 0)
    FP.train                =     sum(y.hat.train[y.train == 0] == 1)
    TP.train                =     sum(y.train[y.hat.train == 1] == 1)
    FPR.train.rf[i]         =     FP.train/N.train
    TPR.train.rf[i]         =     TP.train/P.train
    
    y.hat.test              =     ifelse(p.hat.test > th, 1, 0)
    FP.test                 =     sum(y.hat.test[y.test == 0] == 1)
    TP.test                 =     sum(y.test[y.hat.test == 1] == 1)
    FPR.test.rf[i]          =     FP.test/N.test
    TPR.test.rf[i]          =     TP.test/P.test
  }
  auc.train.rf[m]         =     area_under_curve(FPR.train.rf,TPR.train.rf, method = "trapezoid")
  auc.test.rf[m]          =     area_under_curve(FPR.test.rf,TPR.test.rf, method = "trapezoid") 
  cat(sprintf("m=%.0f|Random Forest: auc.train= %.3f| auc.test= %.3f \n\n", 
              m,auc.train.rf[m],auc.test.rf[m]))
}



# Box Plot
par(mfrow = c(1,2))
boxplot(auc.train.rd,auc.train.ls,auc.train.en, auc.train.rf,main = "Train AUCs",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RD","LS","EN","RF"),
        col = "orange",horizontal = FALSE)

boxplot(auc.test.rd,auc.test.ls,auc.test.en, auc.test.rf, main = "Test AUCs",
        font.main = 2, cex.main = 1, at = c(1,2,3,4), names = c("RD","LS","EN","RF"),
        col = "orange",horizontal = FALSE)

graphics.off()

# CV plot
plot(ridge.cv, sub = "Ridge", cex.sub = 1)
plot(lasso.cv, sub = "Lasso", cex.sub = 1)
plot(elnet.cv, sub = "Elastic Net", cex.sub = 1)

# Time elapsed for CV 
cv_time     =  data.frame(cv_time_rd, cv_time_ls, cv_time_en)

# 90% AUC
quantile(auc.test.rd, c(.05,.95))
quantile(auc.test.ls, c(.05,.95))
quantile(auc.test.en, c(.05,.95))
quantile(auc.test.rf, c(.05,.95))

# Use all the data to run the models

# Ridge
ridge.time.start     =     Sys.time()
ridge.cv.all         =     cv.glmnet(X, y, family = "binomial", alpha = 0,  intercept = TRUE,   nfolds = 10, type.measure="auc")
ridge.all            =     glmnet(X, y, lambda = ridge.cv.all$lambda[which.max(ridge.cv.all$cvm)], family = "binomial", alpha = 0,  intercept = TRUE)
ridge.time.end       =     Sys.time()
time.rd              =     ridge.time.end - ridge.time.start

# Lasso
lasso.time.start     =     Sys.time()
lasso.cv.all         =     cv.glmnet(X, y, family = "binomial", alpha = 1,  intercept = TRUE,  nfolds = 10, type.measure="auc")
lasso.all            =     glmnet(X, y, lambda = lasso.cv.all$lambda[which.max(lasso.cv.all$cvm)], family = "binomial", alpha = 1,  intercept = TRUE)
lasso.time.end       =     Sys.time()
time.ls              =     lasso.time.end - lasso.time.start

# Elnet
elnet.time.start     =     Sys.time()
elnet.cv.all         =     cv.glmnet(X, y, family = "binomial", alpha = 1,  intercept = TRUE,  nfolds = 10, type.measure="auc")
elnet.all            =     glmnet(X, y, lambda = elnet.cv.all$lambda[which.max(elnet.cv.all$cvm)], family = "binomial", alpha = 1,  intercept = TRUE)
elnet.time.end       =     Sys.time()
time.en              =     elnet.time.end - elnet.time.start

# RF
rf.time.start        =     Sys.time()
rf.all               =     randomForest(X,y, mtry = sqrt(p.train), importance = TRUE)
rf.time.end          =     Sys.time()
time.rf              =     rf.time.end - rf.time.start

time2                =     data.frame(time.rd, time.ls, time.en, time.rf)

# Bar-plots of the coefficients
features          = colnames(spambase[-p])

beta.en           = data.frame(features,as.vector(elnet.all$beta))
colnames(beta.en) = c("features","beta")

beta.rd           = data.frame(features,as.vector(ridge.all$beta))
colnames(beta.rd) = c("features","beta")

beta.ls           = data.frame(features,as.vector(lasso.all$beta))
colnames(beta.ls) = c("features","beta")

beta.rf           = data.frame(features,as.vector(rf.all$importance[,3]))
colnames(beta.rf) = c("features","beta")

beta.en$features     =  factor(beta.en$features, levels = beta.en$feature[order(beta.en$beta, decreasing = TRUE)])
beta.rd$features     =  factor(beta.rd$features, levels = beta.rd$feature[order(beta.en$beta, decreasing = TRUE)])
beta.ls$features     =  factor(beta.ls$features, levels = beta.ls$feature[order(beta.en$beta, decreasing = TRUE)])
beta.rf$features     =  factor(beta.rf$features, levels = beta.rf$feature[order(beta.en$beta, decreasing = TRUE)])


# Plot the coefficients

beta.en$method = "Elastic Net"
beta.rf$method = "Random Forest"
beta.ls$method = "Lasso"
beta.rd$method = "Ridge"

betaS        = rbind(beta.en,beta.rd, beta.ls, beta.rf)
betaS$method = factor(betaS$method, levels = c("Elastic Net","Lasso","Ridge","Random Forest"))

ggplot(betaS, aes(x = features, y = beta)) + geom_col() + facet_grid(method~., scales = "free_y") + 
  theme(axis.text.x = element_text(angle = 90, size = 5), axis.ticks.y = element_line())
