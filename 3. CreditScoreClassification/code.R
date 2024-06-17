# install.packages("mlbench")
# install.packages("tree")
# install.packages("e1071")
# install.packages("ramify")
# install.packages("kknn")
# install.packages('randomForest')
# install.packages('glmnet')
# install.packages('tidyr')
# install.packages('xgboost')
# install.packages('lightgbm')

library(tree)
library(mlbench)
library(nnet)
library(ramify)
library(kknn)
library(class)
library(ggplot2)
library(randomForest)
library(glmnet)
library(dplyr)
library(tidyr)
library(xgboost)
library(lightgbm)
library(tidyverse)

#################### #################### #################### #################### ####################
#########################################       Setting        #########################################
#################### #################### #################### #################### #################### 
# rm(list = ls())

test.before = c('mode','rf_test')
train.before = c('mode', 'drop', 'rf_train')
random = c('random_o', 'random_x')
r1 = c('1->1','mode', 'random_o')
r2 = c('1->1','mode', 'random_x')
r3 = c('1->1','rf_test', 'random_o')
r4 = c('1->1','rf_test', 'random_x')

data.param = as.data.frame(crossing(train.before, test.before,random))
data.param = rbind(data.param, r1, r2, r3, r4)

#Option (LR: alpha, lambda)
alpha.vec= seq(0, 1, length.out = 5)               #ridge=0, 반반 lasso=1 
lambda.vec = seq(0, 0.1, length.out = 11)               #penalty
lr.param = crossing(lambda.vec, alpha.vec)
lr.param$comb = NULL
for(i in 1:nrow(lr.param)){
  lr.param$comb[i] = paste('lambda:',lr.param[i,'lambda.vec'],' alpha:', lr.param[i,'alpha.vec'], sep = '')
}

#Option (KNN: k 개수)
neighbor.vec=10:15

#Option (RF)
ntree = c(200,400)
mtry = c(20,30)
nodesize=c(200) 
rf.param = crossing(ntree, mtry, nodesize)
rf.param$comb = NULL
for(i in 1:nrow(rf.param)){
  rf.param$comb[i] = paste('ntree:',rf.param[i, 'ntree'],' mtry:', rf.param[i, 'mtry'],' nodesize', rf.param[i, 'nodesize'], sep = '')
}


#Option XGboost
eta.vec=c(0.01) # eta가 작을 때 정확도 좋았음
max_depth.vec=c(3,5) # eta가 클 때 depth는 클수록 정확도 높았음
early.stop.round.vec=c(150) # 크게 영향이 없는 듯
nrounds.vec=c(200,500) # 작을수록 좋은 듯
xgb.param = crossing(eta.vec, max_depth.vec, early.stop.round.vec, nrounds.vec)
xgb.param$comb = NULL
for(i in 1:nrow(xgb.param)){
  xgb.param$comb[i] = paste('eta:',xgb.param[i, 'eta.vec'],' depth:', xgb.param[i, 'max_depth.vec'],' earlystop:', xgb.param[i, 'early.stop.round.vec'],' nrounds:', xgb.param[i, 'nrounds.vec'] , sep = '')
}

#Option lightGBM
eta.vec=c(0.01) 
max_depth.vec=c(-1,3,5) 
early.stop.round.vec=c(150) 
nrounds.vec=c(200,500) 
lgb.param = crossing(eta.vec, max_depth.vec, early.stop.round.vec, nrounds.vec)
lgb.param$comb = NULL
for(i in 1:nrow(lgb.param)){
  lgb.param$comb[i] = paste('eta:',lgb.param[i, 'eta.vec'],' depth:', lgb.param[i, 'max_depth.vec'],' earlystop:', lgb.param[i, 'early.stop.round.vec'],' nrounds:', lgb.param[i, 'nrounds.vec'] , sep = '')
}


# Option ensemble
m1 = c( c( (0:9) / 10),1/3 , 1/4);
m2 = c( c( (0:9) / 10),1/3 , 1/4);
m3 = c( c( (0:9) / 10),1/3 , 1/4);
m4 = c( c( (0:9) / 10),1/3 , 1/4);

s = crossing(m1,m2,m3,m4)
s = s %>% as.matrix()
weight.vec = NULL
for(num in 1:nrow(s)){
  if(sum(s[num,]) ==1){
    # print(s[num,])
    weight.vec = rbind(weight.vec, s[num,] )
  }
}

#model2 측정 지표
num_model=7 + nrow(weight.vec)
acc.mat=sen.mat=spc.mat=fixed_acc.mat=ratio.mat=matrix(NA, 100, num_model)
param.opt.mat = matrix(NA, 100, 5)

#model3 측정 지표
num_model=7 + nrow(weight.vec)
month.acc.mat=month.sen.mat=month.spc.mat=month.fixed_acc.mat=month.ratio.mat=matrix(NA, num_model*8, 100)

#################### #################### #################### #################### ####################
#################### ###################        함수저장       #################### ####################
#################### #################### #################### #################### #################### 

accuracy.func=function(table){
  acc=sum(diag(table))/sum(table)
  table[3, 1]=5*table[3, 1]   #Bad를 Good으로 판단하는 경우 (*5)
  fixed_acc=sum(diag(table))/sum(table)
  
  return(c(acc, fixed_acc))
}

specificity.func=function(table){
  macro.avg=((table[2, 2]+table[2, 3]+table[3, 2]+table[3, 3])/(sum(table)-sum(table[, 1]))+
               (table[1, 1]+table[1, 3]+table[3, 1]+table[3, 3])/(sum(table)-sum(table[, 2]))+
               (table[1, 1]+table[1, 2]+table[2, 1]+table[2, 2])/(sum(table)-sum(table[, 3])))/3
  return(macro.avg)
}

sensitivity.func=function(table){
  macro.avg=(table[1, 1]/sum(table[, 1])+
               table[2, 2]/sum(table[, 2])+table[3, 3]/sum(table[, 3]))/3
  return(macro.avg)
}

ratio.fun = function(table){
  (table[2,1] + table[3,1] + table[3,2])/(sum(table) - sum(diag(table)))
}

# knn.january.func=function(train, test, neighbor_opt=1){
#   new_train=subset(train, select=-c(Customer_ID, Credit_Score, Before_Credit_Score))
#   new_test=subset(test, select=-c(Customer_ID, Before_Credit_Score,Credit_Score))
#   pred=knn(new_train, new_test, train$Before_Credit_Score, k=neighbor_opt)
#   return(pred)
# }

random.fun = function(X_df, rep_num){
  mask=NULL
  for(sample_iter in 1:nrow(X_df)/rep_num){
    sub_mask=rep(0, rep_num)
    sub_mask[sample(1:rep_num, size=2)]=1
    mask=c(mask, sub_mask)
  }
  
  for(iter_row in 1:nrow(X_df)){
    if(mask[iter_row]==1){
      old_Before_Credit_Score=X_df[iter_row, ]$Before_Credit_Score
      sample.vec=1:3; sample.vec=sample.vec[sample.vec!=old_Before_Credit_Score]
      X_df[iter_row, ]$Before_Credit_Score=sample(sample.vec, 1)
    }
  }
  return(X_df)
}

drop.fun = function(train){
  train = train %>% filter(Month != 1)
  return(train)
}

rf.january.func = function(train, test, opt, ntree = 400, mtry = 13, nodesize = 150){
  
  if(opt=="rf_test"){
    new_train=subset(train, select=-c(Customer_ID, Before_Credit_Score))
    new_test=subset(test, select=-c(Customer_ID, Before_Credit_Score))
    
    rf_fit=randomForest(new_train$Credit_Score~., data = new_train, ntree = ntree,
                        mtry = mtry, nodesize= nodesize)
    for(iter_row in 1:nrow(test)){
      if(test[iter_row, ]$Month==1){
        rf.pred.prob = predict(rf_fit,test[iter_row, ], type = 'prob')
        rf.pred = apply(rf.pred.prob,1,which.max)
        test[iter_row, ]$Before_Credit_Score=rf.pred
      }
    }
  }else if(opt=='rf_train'){
    new_train=subset(train, select=-c(Customer_ID, Before_Credit_Score))%>%filter(Month!=1)
    new_test=subset(test, select=-c(Customer_ID, Before_Credit_Score))%>%filter(Month==1)
    
    rf_fit=randomForest(new_train$Credit_Score~., data = new_train, ntree = ntree,
                        mtry = mtry, nodesize= nodesize)
    for(iter_row in 1:nrow(test)){
      if(test[iter_row, ]$Month==1){
        rf.pred.prob = predict(rf_fit,test[iter_row, ], type = 'prob')
        rf.pred = apply(rf.pred.prob,1,which.max)
        test[iter_row, ]$Before_Credit_Score=rf.pred
      }
    }
    
    y_true_df = test %>% filter(Month == 1)
    y_true = y_true_df$Credit_Score
    y_pred_df= test %>% filter(Month ==1)
    y_pred =y_pred_df$Before_Credit_Score
    tab = table(y_pred, y_true)
    acc = accuracy.func(tab)[1]
  }
  return(test)
}

month.compare.func=function(y_pred, y_true = test$Credit_Score){
  res.acc=matrix(NA, 8, 5)
  for(i in 1:8){
    mask=seq(i, length(y_pred), by=8)  
    
    y_pred_sam=y_pred[mask]
    y_true_sam=y_true[mask]
    y_pred_sam = factor(y_pred_sam, c(1,2,3))
    y_true_sam = factor(y_true_sam, c(1,2,3))

    sam_table=table(y_pred_sam, y_true_sam)
    
    res.acc[i,1]=accuracy.func(sam_table)[1]
    res.acc[i,2]=sensitivity.func(sam_table)
    res.acc[i,3]=specificity.func(sam_table)
    res.acc[i,4]=accuracy.func(sam_table)[2]
    res.acc[i,5]=ratio.fun(sam_table)
  }
  return(res.acc)
}

make.train.stack=function(train, n_folds=5){
  unique_Customer_ID=unique(train$Customer_ID)
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  
  sampled_unique_Customer_ID=unique_Customer_ID[nset]
  train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
  return(train)
}

#################### ####################      elastic         #################### ####################
#LR tun
lr.tun = function(train, test, param = lr.param , n_folds = 5){
  unique_Customer_ID=unique(train$Customer_ID)
  unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID))   #랜덤 shuffle
  folder_interval = length(unique_Customer_ID) / n_folds
  valid.acc.mat=matrix(NA, n_folds, nrow(lr.param))
  nset = c(1:folder_interval)
  
  for(iiter in 1:nrow(lr.param)) {
    nset = c(1:folder_interval)
    for(iiiter in 1:n_folds){   
      print(c(iiter, iiiter))
      
      sampled_unique_Customer_ID=unique_Customer_ID[nset]
      new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
      new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
      
      new_train=subset(new_train, select=-(Customer_ID))
      new_test=subset(new_test, select=-(Customer_ID))
      
      alp = as.numeric(lr.param[iiter,'alpha.vec'])
      lam = as.numeric(lr.param[iiter,'lambda.vec'])
    
      mm=model.matrix(Credit_Score~. -1, new_train)
      
      glmnet.fit=glmnet(mm, new_train$Credit_Score, family = 'multinomial',
                        alpha= alp, lambda = lam)
      
      pred=argmax(predict(glmnet.fit,
                          model.matrix(Credit_Score ~. -1, new_test),
                          type = 'response'))
      
      pred=factor(pred, levels = c(1,2,3))
      
      lr.table.v=table(pred, new_test$Credit_Score)
      
      valid.acc.mat[iiiter, iiter]=accuracy.func(lr.table.v)[1]
      
      
      nset = nset + folder_interval
    }
  }
  df_acc=apply(valid.acc.mat, 2, mean)
  return(df_acc)
}

lr.stack.opt=function(train, alpha_opt, lambda_opt, n_folds=5){
  unique_Customer_ID=unique(train$Customer_ID)
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  sampled_unique_Customer_ID=unique_Customer_ID[nset]
  new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
  new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
  
  train=subset(new_train, select=-(Customer_ID))
  holdout=subset(new_test, select=-(Customer_ID))
  
  stack_y = holdout$Credit_Score
  
  mm=model.matrix(Credit_Score~. -1, train)
  
  glmnet.fit=glmnet(mm, train$Credit_Score, family='multinomial',
                    alpha = alpha_opt, lambda = lambda_opt)
  
  pred=argmax(predict(glmnet.fit,
                      model.matrix(Credit_Score ~. -1, holdout),
                      type = 'response'))
  
  pred=factor(pred, levels = c(1,2,3))
  return(list(pred, stack_y))
}


lr.optimal.func = function(train, test, alpha_opt, lambda_opt){
  train=subset(train, select=-(Customer_ID))
  X_test=subset(test, select=-(Customer_ID))
  
  mm=model.matrix(Credit_Score~. -1, train)
  
  glmnet.fit=glmnet(mm, train$Credit_Score, family='multinomial',
                    alpha = alpha_opt, lambda = lambda_opt)
  
  
  lr_pred_prob=matrix(NA, nrow(X_test), 3)
  pred=rep(NA, nrow(X_test))
  for(iter in 1:nrow(X_test)){
    if(iter %%8 == 1){
      mmt=model.matrix(Credit_Score~. -1, X_test[iter, ])
      lr_pred_prob[iter, ]=predict(glmnet.fit, mmt, type = 'response')
      pred[iter]=which.max(lr_pred_prob[iter, ])
    }else{
      X_test[iter,]$Before_Credit_Score=pred[iter-1]
      lr_pred_prob[iter, ]=predict(glmnet.fit, model.matrix(Credit_Score~. -1, X_test[iter, ]), type = 'response')
      pred[iter]=which.max(lr_pred_prob[iter, ])
    }
  }
  pred=factor(pred, levels = c(1,2,3))
  lr.table=table(pred, test$Credit_Score)
  
  res.acc=rep(NA, 5)
  res.acc[1]=accuracy.func(lr.table)[1]
  res.acc[2]=sensitivity.func(lr.table)
  res.acc[3]=specificity.func(lr.table)
  res.acc[4]=accuracy.func(lr.table)[2]  #fixed_acc
  res.acc[5]=ratio.fun(lr.table)
  return(list(res.acc, lr_pred_prob))
}


#################### ####################      KNN         #################### ####################
#KNN Neighbor 수
knn.tun=function(train, neighbor.vec, n_folds=5){
  
  unique_Customer_ID=unique(train$Customer_ID)
  unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID))   #랜덤 shuffle
  folder_interval = length(unique_Customer_ID) / n_folds
  valid_neighbor.acc.mat=matrix(NA, n_folds, length(neighbor.vec))
  nset = c(1:folder_interval)
  
  for(iiter in 1:length(neighbor.vec)){    
    nset = c(1:folder_interval)
    for(iiiter in 1:n_folds){   
      print(c(iiter, iiiter))
      
      sampled_unique_Customer_ID=unique_Customer_ID[nset]
      new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
      new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
      
      new_X_train=subset(new_train, select=-c(Credit_Score, Customer_ID))
      new_X_test=subset(new_test, select=-c(Credit_Score, Customer_ID))
      
      pred=knn(new_X_train, new_X_test, new_train$Credit_Score, k=neighbor.vec[iiter])
      knn_table_v=table(pred, new_test$Credit_Score)
      
      valid_neighbor.acc.mat[iiiter, iiter]=accuracy.func(knn_table_v)[1]
      
      nset = nset + folder_interval
    }
  }
  df_acc=apply(valid_neighbor.acc.mat, 2, mean)
  return(df_acc)
}

#Optimal KNN
knn.optimal.func=function(train, test, neighbor = neighbor_opt){
  X_train=subset(train, select=-c(Credit_Score, Customer_ID))
  X_test=subset(test, select=-c(Credit_Score, Customer_ID))
  
  pred=rep(NA, nrow(X_test))
  for(iter in 1:nrow(X_test)){
    if(iter%%8==1){
      pred[iter]=knn(X_train, X_test[iter,], train$Credit_Score, k = neighbor)
    }else{
      X_test[iter,]$Before_Credit_Score=pred[iter-1]
      pred[iter]=knn(X_train, X_test[iter,], train$Credit_Score, k = neighbor)
    }
  }
  pred=factor(pred, levels = c(1,2,3))
  knn_table=table(pred, test$Credit_Score)
  res.acc=rep(NA, 4)
  res.acc[1]=accuracy.func(knn_table)[1]
  res.acc[2]=sensitivity.func(knn_table)
  res.acc[3]=specificity.func(knn_table)
  res.acc[4]=accuracy.func(knn_table)[2]
  return(res.acc)
}


#################### ####################      TREE          #################### ####################
#RF
tree.tun = function(train,test,n_folds = 5,rf.param ){
  
  unique_Customer_ID=unique(train$Customer_ID)
  unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID))   #랜덤 shuffle
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  
  rf_validation.acc.mat=matrix(NA, n_folds, nrow(rf.param))
  
  for(iiter in 1:nrow(rf.param)){      
    nset = c(1:folder_interval)
    for(iiiter in 1:n_folds){
      print(c(iiter, iiiter))
      
      sampled_unique_Customer_ID=unique_Customer_ID[nset]
      new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
      new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
      
      new_train=subset(new_train, select=-(Customer_ID))
      new_test=subset(new_test, select=-(Customer_ID))
      
      rf_fit=randomForest(new_train$Credit_Score~., data = new_train, ntree=as.integer(rf.param[iiter,'ntree']) , mtry =as.integer(rf.param[iiter,'mtry']) , nodesize=as.integer(rf.param[iiter,'nodesize']))
      
      rf_pred_val = predict(rf_fit,new_test)  
      rf_table_v=table(rf_pred_val, new_test$Credit_Score)
      rf_validation.acc.mat[iiiter, iiter]=accuracy.func(rf_table_v)[1]
      nset = nset + folder_interval
    }
  }
  ret.mat = (apply(rf_validation.acc.mat, 2, mean))
  return (ret.mat)
}

tree.stack.opt=function(train, ntree, mtry, nodesize, n_folds=5){
  unique_Customer_ID=unique(train$Customer_ID)
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  sampled_unique_Customer_ID=unique_Customer_ID[nset]
  new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
  new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
  
  train=subset(new_train, select=-(Customer_ID))
  holdout=subset(new_test, select=-(Customer_ID))
  
  rf_fit=randomForest(train$Credit_Score~., data = train, ntree=ntree , mtry = mtry , nodesize= nodesize)
  rf_pred_val = predict(rf_fit,holdout)  
  rf_pred_val = factor(rf_pred_val, levels = c(1,2,3))
  return(rf_pred_val)
}

# RF opt
tree.optimal = function(train,test,ntree,mtry,nodesize){
  train=subset(train, select=-(Customer_ID))
  X_test=subset(test, select=-(Customer_ID))
  
  rf_fit = randomForest(train$Credit_Score~. , data = train, ntree= ntree, mtry = mtry, nodesize= nodesize,importance = T)
  
  rf_fit$importance
  
  rf_pred_prob = matrix(NA, nrow(X_test), 3)
  rf_pred = rep(NA, nrow(X_test))
  for(iter in 1:nrow(X_test)){
    if(iter %%8 == 1){
      rf_pred_prob[iter, ]=predict(rf_fit, X_test[iter, ], type = 'prob')
      rf_pred[iter]=which.max(rf_pred_prob[iter, ])
    }else{
      X_test[iter,]$Before_Credit_Score=rf_pred[iter-1]
      rf_pred_prob[iter, ]=predict(rf_fit, X_test[iter, ], type = 'prob')
      rf_pred[iter]=which.max(rf_pred_prob[iter, ])
    }
  }
  rf_pred=factor(rf_pred, levels = c(1,2,3))
  rf_table = table(rf_pred, test$Credit_Score)
  rf.acc = rep(NA, 5)
  rf.acc[1]=accuracy.func(rf_table)[1]
  rf.acc[2]=sensitivity.func(rf_table)
  rf.acc[3]=specificity.func(rf_table)
  rf.acc[4]=accuracy.func(rf_table)[2]
  rf.acc[5]=ratio.fun(rf_table)
  return(list(rf.acc,rf_pred_prob))
  
}

#################### ####################      XGBoost    #################### ####################
#XGB tun
xgb.tun=function(train, param=xgb.param, n_folds=5){
  
  train$Before_Credit_Score=as.integer(train$Before_Credit_Score)-1
  train$Before_Credit_Score=factor(train$Before_Credit_Score, levels=c(0, 1, 2))
  
  unique_Customer_ID=unique(train$Customer_ID)
  unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID))
  folder_interval = length(unique_Customer_ID) / n_folds
  xgb_validation.acc.mat=matrix(NA, n_folds, nrow(xgb.param))
  nset = c(1:folder_interval)
  
  
  for(iiter in 1:nrow(xgb.param)){    
    nset = c(1:folder_interval)
    for(iiiter in 1:n_folds){   
      print(c(iiter, iiiter))
      
      eta=as.numeric(xgb.param[iiter,'eta.vec'])
      max_depth=as.numeric(xgb.param[iiter,'max_depth.vec'])
      early.stop.round=as.numeric(xgb.param[iiter,'early.stop.round.vec'])
      nrounds=as.numeric(xgb.param[iiter,'nrounds.vec'])
      
      sampled_unique_Customer_ID=unique_Customer_ID[nset]
      new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
      new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
      
      new_train=subset(new_train, select=-(Customer_ID))
      new_test=subset(new_test, select=-(Customer_ID))
      
      new_y_train=as.integer(new_train$Credit_Score)-1
      new_y_test=as.integer(new_test$Credit_Score)-1
      new_X_train=model.matrix(Credit_Score~. -1, new_train)
      new_X_test=model.matrix(Credit_Score~. -1, new_test)
      
      xgboost_train=xgb.DMatrix(new_X_train, label=as.matrix(new_y_train))
      xgboost_test=xgb.DMatrix(new_X_test, label=as.matrix(new_y_test))
      
      xgb_fit=xgboost(data=xgboost_train, eta=eta, max.depth=max_depth, objective='multi:softprob', num_class=3, early_stopping_rounds=early.stop.round, nrounds=nrounds, verbose=0)                              
      xgb_pred=predict(xgb_fit, xgboost_test, reshape=T)
      xgb_pred=as.data.frame(xgb_pred)
      
      xgb_pred_val=apply(xgb_pred, 1, which.max)
      xgb_pred_val = factor(xgb_pred_val, c(1,2,3))
      
      new_y_test_lab=as.integer(new_y_test)+1
      
      xgb_table_v=table(xgb_pred_val, new_y_test_lab)
      
      xgb_validation.acc.mat[iiiter, iiter]=accuracy.func(xgb_table_v)[1]   #acc(일단 그냥)
      nset=nset+folder_interval
      
    }
  }
  ret.mat=apply(xgb_validation.acc.mat, 2, mean)
  return(ret.mat)
}

xgb.stacking.func=function(train, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt, n_folds=5){
  train$Before_Credit_Score=as.integer(train$Before_Credit_Score)-1
  train$Before_Credit_Score=factor(train$Before_Credit_Score, levels=c(0, 1, 2))
  
  unique_Customer_ID=unique(train$Customer_ID)
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  
  sampled_unique_Customer_ID=unique_Customer_ID[nset]
  new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
  new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
  
  train=subset(new_train, select=-(Customer_ID))
  holdout=subset(new_test, select=-(Customer_ID))
  
  new_y_train=as.integer(new_train$Credit_Score)-1
  new_y_holdout=as.integer(holdout$Credit_Score)-1
  new_X_train=model.matrix(Credit_Score~. -1, train)
  new_X_holdout=model.matrix(Credit_Score~. -1, holdout)
  
  xgboost_train=xgb.DMatrix(new_X_train, label=as.matrix(new_y_train))
  xgboost_test=xgb.DMatrix(new_X_holdout, label=as.matrix(new_y_holdout))
  
  xgb_fit=xgboost(data=xgboost_train, eta=eta_opt, max.depth=max_depth_opt, objective='multi:softprob', num_class=3, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt, verbose=0)                              
  xgb_pred=predict(xgb_fit, xgboost_test, reshape=T)
  xgb_pred=as.data.frame(xgb_pred)
  
  pred=apply(xgb_pred, 1, which.max)
  pred=factor(pred, levels = c(1,2,3))
  return(pred)
}

#Optimal XGB
xgb.optimal.func = function(train, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt){
  
  new_train=subset(train, select=-(Customer_ID))
  new_test=subset(test, select=-(Customer_ID))
  
  new_train$Before_Credit_Score=as.integer(new_train$Before_Credit_Score)-1
  new_train$Before_Credit_Score=factor(new_train$Before_Credit_Score, levels=c(0, 1, 2))
  
  new_test$Before_Credit_Score=as.integer(new_test$Before_Credit_Score)-1
  new_test$Before_Credit_Score=factor(new_test$Before_Credit_Score, levels=c(0, 1, 2))
  
  y_train=as.integer(new_train$Credit_Score)-1
  y_test=as.integer(new_test$Credit_Score)-1
  X_train=model.matrix(Credit_Score~. -1, new_train)
  xgboost_train=xgb.DMatrix(X_train, label=as.matrix(y_train))
  
  xgb_fit=xgboost(data=xgboost_train, eta=eta, max.depth=max_depth, objective='multi:softprob', num_class=3, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt, verbose=0)                              
  xgb_pred=NULL
  pred=rep(NA, nrow(new_test))
  
  for(iter in 1:nrow(new_test)){
    if(iter%%8==1){
      new_test[iter, ]$Before_Credit_Score=as.integer(new_test[iter, ]$Before_Credit_Score)-1
      X_test=model.matrix(Credit_Score~. -1, new_test[iter, ])
      xgboost_test=xgb.DMatrix(X_test, label=as.matrix(y_test[iter], nrow=1))
      temp=predict(xgb_fit, xgboost_test, reshape=T)
      xgb_pred=rbind(xgb_pred, temp)
      pred[iter]=which.max(xgb_pred[iter, ])-1
    }else{
      new_test[iter,]$Before_Credit_Score=pred[iter-1]
      X_test=model.matrix(Credit_Score~. -1, new_test[iter, ])
      xgboost_test=xgb.DMatrix(X_test, label=as.matrix(y_test[iter], nrow=1))
      temp=predict(xgb_fit, xgboost_test, reshape=T)
      xgb_pred=rbind(xgb_pred, temp)
      pred[iter]=which.max(xgb_pred[iter, ])-1
    }
  }
  
  xgb_pred=as.data.frame(xgb_pred)
  xgb_pred_val=apply(xgb_pred, 1, which.max)
  y_test_lab=as.integer(y_test)+1
  
  xgb_pred_val=factor(xgb_pred_val, levels = c(1,2,3))
  xgb_table=table(xgb_pred_val, y_test_lab)
  
  res.acc=rep(NA, 5)
  res.acc[1]=accuracy.func(xgb_table)[1]  #acc
  res.acc[2]=sensitivity.func(xgb_table)
  res.acc[3]=specificity.func(xgb_table)
  res.acc[4]=accuracy.func(xgb_table)[2]  #fixed_acc
  res.acc[5]=ratio.fun(xgb_table)
  return(list(res.acc, xgb_pred))
}


#################### ####################      LGBM    #################### ####################
lgbm.tun=function(train, test, param=lgb.param, n_folds=5){
  
  train$Before_Credit_Score = as.integer(train$Before_Credit_Score) - 1
  train$Before_Credit_Score = factor(train$Before_Credit_Score, c(0,1,2))
  
  unique_Customer_ID=unique(train$Customer_ID)
  unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID)) # random shuffle
  folder_interval = length(unique_Customer_ID) / n_folds
  lgbm_validation.acc.mat=matrix(NA, n_folds, nrow(lgb.param))
  nset = c(1:folder_interval)
  
  for(iiter in 1:nrow(lgb.param)){    
    nset = c(1:folder_interval)
    
    for(iiiter in 1:n_folds){   
      print(c(iiter, iiiter))
      
      lgb_eta=as.numeric(lgb.param[iiter,'eta.vec'])
      lgb_max_depth=as.numeric(lgb.param[iiter,'max_depth.vec'])
      lgb_early.stop.round=as.numeric(lgb.param[iiter,'early.stop.round.vec'])
      lgb_nrounds=as.numeric(lgb.param[iiter,'nrounds.vec'])
      
      params = list( objective = "multiclass",
                     metric = "multi_error",
                     num_class = 3,
                     num_iterations = lgb_nrounds,
                     max_depth = lgb_max_depth,
                     eta = lgb_eta,
                     early_stopping_rounds =lgb_early.stop.round
      )
      
      sampled_unique_Customer_ID=unique_Customer_ID[nset]
      
      new_test=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
      new_train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
      
      lgb_set = (unique_Customer_ID[-nset])
      
      lgb_sample = sample(1:length(lgb_set), length(lgb_set) * 0.8)
      
      lgb_train_sample = lgb_set[lgb_sample]
      
      lgb_test_sample = lgb_set[-lgb_sample]
      
      
      lgb_train = subset(new_train, (Customer_ID %in% lgb_train_sample))
      
      lgb_val = subset(new_train, (Customer_ID %in% lgb_test_sample))
      
      
      lgb_train= subset(lgb_train, select=-(Customer_ID))
      lgb_val= subset(lgb_val, select=-(Customer_ID))
      new_test =subset(new_test, select=-(Customer_ID))
      
      lgb_X_train = subset(lgb_train, select=-Credit_Score)
      lgb_y_train=as.integer(lgb_train$Credit_Score) -1
      
      lgb_X_val=subset(lgb_val, select=-Credit_Score)
      lgb_y_val=as.integer(lgb_val$Credit_Score) -1
      
      lgb_X_test=subset(new_test, select=-Credit_Score)
      lgb_y_test= as.integer(new_test$Credit_Score)-1
      
      
      dtrain = lgb.Dataset(data = as.matrix(lgb_X_train), label = as.matrix(lgb_y_train))
      
      dval = lgb.Dataset.create.valid(dataset = dtrain, data = as.matrix(lgb_X_val), label = as.matrix(lgb_y_val))
      dval = list(test = dval)
      
      lgb_fit = lgb.train(
        param =params,
        data =dtrain,
        valids = dval,
        verbose=-1
      )
      lgb_pred = predict(lgb_fit, as.matrix(lgb_X_test))
      
      lgb_pred.mat= matrix(lgb_pred, 3,length(lgb_pred)/3)
      lgb_pred_val = apply(lgb_pred.mat,2,which.max) - 1;  #y_pred.vec
      lgb_pred_val = factor(lgb_pred_val, c(0,1,2))
      
      lgb_table_v=table(lgb_pred_val, lgb_y_test)
      
      lgbm_validation.acc.mat[iiiter, iiter]=accuracy.func(lgb_table_v)[1]
      nset=nset+folder_interval
      
    }
  }
  
  
  ret.mat=(apply(lgbm_validation.acc.mat, 2, mean))
  return(ret.mat)
}

#LGBM stacking
lgbm.stacking.func=function(train, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt, n_folds=5){
  
  train$Before_Credit_Score=as.integer(train$Before_Credit_Score)-1
  train$Before_Credit_Score=factor(train$Before_Credit_Score, levels=c(0, 1, 2))
  
  unique_Customer_ID=unique(train$Customer_ID)
  folder_interval = length(unique_Customer_ID) / n_folds
  nset = c(1:folder_interval)
  
  sampled_unique_Customer_ID=unique_Customer_ID[nset]
  holdout=subset(train, (Customer_ID %in% sampled_unique_Customer_ID))
  train=subset(train, !(Customer_ID %in% sampled_unique_Customer_ID))
  
  lgb_set=(unique_Customer_ID[-nset])
  
  lgb_sample=sample(1:length(lgb_set), length(lgb_set)*0.8)
  lgb_train_sample=lgb_set[lgb_sample]
  lgb_test_sample=lgb_set[-lgb_sample]
  
  
  lgb_train=subset(train, (Customer_ID %in% lgb_train_sample))
  lgb_val=subset(train, (Customer_ID %in% lgb_test_sample))
  
  lgb_train= subset(lgb_train, select=-(Customer_ID))
  lgb_val= subset(lgb_val, select=-(Customer_ID))
  holdout=subset(holdout, select=-(Customer_ID))
  
  lgb_X_train = subset(lgb_train, select=-Credit_Score)
  lgb_y_train=as.integer(lgb_train$Credit_Score) -1
  
  lgb_X_val=subset(lgb_val, select=-Credit_Score)
  lgb_y_val=as.integer(lgb_val$Credit_Score) -1
  
  lgb_X_holdout=subset(holdout, select=-Credit_Score)
  lgb_y_holdout=as.integer(holdout$Credit_Score)-1
  
  dtrain=lgb.Dataset(data=as.matrix(lgb_X_train), label=as.matrix(lgb_y_train))
  dval=lgb.Dataset.create.valid(dataset=dtrain, data=as.matrix(lgb_X_val), label=as.matrix(lgb_y_val))
  dval=list(test=dval)
  
  params = list( objective = "multiclass",
                 metric = "multi_error",
                 num_class = 3,
                 num_iterations = nrounds_opt,
                 max_depth = max_depth_opt,
                 eta = eta_opt,
                 early_stopping_rounds =early.stop.round_opt
  )
  
  lgb_fit=lgb.train(
    param=params,
    data=dtrain,
    valids=dval,
    verbose=-1
  )
  lgb_pred = predict(lgb_fit, as.matrix(lgb_X_holdout))
  
  lgb_pred.mat= matrix(lgb_pred, 3,length(lgb_pred)/3)
  pred=apply(lgb_pred.mat,2,which.max)
  pred=factor(pred, levels = c(1,2,3))
  return(pred)
}

#Optimal LGBM
lgbm.optimal.func = function(train, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt){
  
  train$Before_Credit_Score = as.integer(train$Before_Credit_Score) - 1
  train$Before_Credit_Score = factor(train$Before_Credit_Score, c(0,1,2))
  
  test$Before_Credit_Score = as.integer(test$Before_Credit_Score) - 1
  test$Before_Credit_Score = factor(test$Before_Credit_Score, c(0,1,2))
  
  unique_Customer_ID=unique(train$Customer_ID)
  training_unique_Customer_ID=sample(unique_Customer_ID, length(unique_Customer_ID)*0.7) 
  
  new_val=subset(train, !(Customer_ID %in% training_unique_Customer_ID))
  new_train=subset(train, (Customer_ID %in% training_unique_Customer_ID))
  
  X_train=subset(new_train, select=-Credit_Score)
  y_train=as.integer(new_train$Credit_Score)-1
  
  X_val=subset(new_val, select=-Credit_Score)
  y_val=as.integer(new_val$Credit_Score)-1
  
  X_test=subset(test, select=-Credit_Score)
  y_test=as.integer(test$Credit_Score)-1
  
  X_train= subset(X_train, select=-(Customer_ID))
  X_val= subset(X_val, select=-(Customer_ID))
  X_test =subset(X_test, select=-(Customer_ID))
  
  dtrain = lgb.Dataset(data = as.matrix(X_train), label = as.matrix(y_train))
  
  dval = lgb.Dataset.create.valid(dataset = dtrain, data = as.matrix(X_val), label = as.matrix(y_val))
  dval = list(test = dval)
  
  params = list( objective = "multiclass",
                 metric = "multi_error",
                 num_class = 3,
                 num_iterations = nrounds_opt,
                 max_depth = max_depth_opt,
                 eta = eta_opt,
                 early_stopping_rounds =early.stop.round_opt
  )
  
  lgb_fit = lgb.train(
    param =params,
    data =dtrain,
    valids = dval,
    verbose=-1
  )
  
  lgb_pred.mat = NULL
  pred=rep(NA, nrow(X_test))
  for(pos in 1:nrow(X_test)){
    if(pos%%8==1){
      lgb_pred=predict(lgb_fit, as.matrix(X_test[pos,]))
      lgb_pred.mat=rbind(lgb_pred.mat, lgb_pred)
      pred[pos]=which.max(lgb_pred) - 1
    }else{
      X_test[pos,]$Before_Credit_Score=pred[pos-1]
      lgb_pred=predict(lgb_fit,as.matrix( X_test[pos,]))
      lgb_pred.mat=rbind(lgb_pred.mat, lgb_pred)
      pred[pos]=which.max(lgb_pred) - 1
      
    }
  }
  
  lgb_pred_val = apply(lgb_pred.mat,1,which.max)
  lgb_pred_val = factor(lgb_pred_val, levels = c(1,2,3))
  y_test = as.integer(y_test)+1
  
  lgb_table=table(lgb_pred_val, y_test)
  res.acc=rep(NA, 5)
  res.acc[1]=accuracy.func(lgb_table)[1]
  res.acc[2]=sensitivity.func(lgb_table)
  res.acc[3]=specificity.func(lgb_table)
  res.acc[4]=accuracy.func(lgb_table)[2]
  res.acc[5]=ratio.fun(lgb_table)
  
  return(list(res.acc, lgb_pred.mat))
}

#################### ####################      Ensemble   #################### ####################

ensemble.fun = function(w.vec , m1,m2,m3,m4, Y_TEST ){
  
  ret.mat = matrix(0, ncol = ncol(m1) , nrow = nrow(m1) )
  
  model.vec = list(m1,m2,m3,m4)
  for(c in 1:length(w.vec)){
    if(w.vec[c] != 0){
      ret.mat = ret.mat + w.vec[c] * model.vec[[c]]
    }
  }
  
  ret.mat = apply(ret.mat, 1,which.max)
  ret.mat = factor(ret.mat, c(1,2,3))
  
  ensemble.tab = table(ret.mat,Y_TEST)
  res.acc=rep(NA, 4)
  
  res.acc[1]=accuracy.func(ensemble.tab)[1]
  res.acc[2]=sensitivity.func(ensemble.tab)
  res.acc[3]=specificity.func(ensemble.tab)
  res.acc[4]=accuracy.func(ensemble.tab)[2]
  res.acc[5]=ratio.fun(ensemble.tab)
  return (list(res.acc, ret.mat))
}

#################### ####################      Stacking   #################### ####################
rf.stacking.fun = function(x1 = lr.stack.vec, x2 = tree.stack.vec, x3 = xgb.stack.vec, x4 = lgb.stack.vec, Credit_Score = stack_y,
                           y1 = lr.stack.test.vec, y2 = tree.stack.test.vec, y3 = xgb.stack.test.vec, y4 = lgb.stack.test.vec, test = test, ntree = 300, mtry = 4, nodesize = 150){
  train = as.data.frame(cbind(x1, x2, x3, x4, Credit_Score))  
  
  train$x1 = factor(train$x1, c(1,2,3))
  train$x2 = factor(train$x2, c(1,2,3))
  train$x3 = factor(train$x3, c(1,2,3))
  train$x4 = factor(train$x4, c(1,2,3))
  train$Credit_Score = factor(train$Credit_Score, c(1,2,3))
  
  lr.pred = apply(y1, 1, which.max)
  tree.pred = apply(y2, 1, which.max)
  xgb.pred = apply(y3, 1, which.max)
  lgb.pred = apply(y4, 1, which.max)
  X_test = as.data.frame(cbind(lr.pred, tree.pred, xgb.pred, lgb.pred))
  
  X_test$lr.pred = factor(X_test$lr.pred, c(1,2,3))
  X_test$tree.pred = factor(X_test$tree.pred, c(1,2,3))
  X_test$xgb.pred = factor(X_test$xgb.pred, c(1,2,3))
  X_test$lgb.pred = factor(X_test$lgb.pred, c(1,2,3))
  
  names(X_test) = c('x1', 'x2', 'x3', 'x4')
  
  rf_fit = randomForest(train$Credit_Score~. , data = train, ntree = 300, mtry = 4, nodesize = 150)
  rf_pred = predict(rf_fit, X_test)
  rf_pred = factor(rf_pred, levels = c(1,2,3))
  
  rf_table = table(rf_pred, test$Credit_Score)
  
  res.acc = rep(NA, 4)
  res.acc[1]=accuracy.func(rf_table)[1]
  res.acc[2]=sensitivity.func(rf_table)
  res.acc[3]=specificity.func(rf_table)
  res.acc[4]=accuracy.func(rf_table)[2]
  res.acc[5]=ratio.fun(rf_table)
  
  return(list(res.acc, rf_pred))
}


lr.stacking.fun = function(x1 = lr.stack.vec, x2 = tree.stack.vec, x3 = xgb.stack.vec, x4 = lgb.stack.vec, Credit_Score = stack_y,
                           y1 = lr.stack.test.vec, y2 = tree.stack.test.vec, y3 = xgb.stack.test.vec, y4 = lgb.stack.test.vec, test = test, alpha_opt=0, lambda_opt=0.001){
  train = as.data.frame(cbind(x1, x2, x3, x4, Credit_Score))  
  
  train$x1 = factor(train$x1, c(1,2,3))
  train$x2 = factor(train$x2, c(1,2,3))
  train$x3 = factor(train$x3, c(1,2,3))
  train$x4 = factor(train$x4, c(1,2,3))
  train$Credit_Score = factor(train$Credit_Score, c(1,2,3))
  
  lr.pred = apply(y1, 1, which.max)
  tree.pred = apply(y2, 1, which.max)
  xgb.pred = apply(y3, 1, which.max)
  lgb.pred = apply(y4, 1, which.max)
  X_test = as.data.frame(cbind(lr.pred, tree.pred, xgb.pred, lgb.pred))
  
  X_test$lr.pred = factor(X_test$lr.pred, c(1,2,3))
  X_test$tree.pred = factor(X_test$tree.pred, c(1,2,3))
  X_test$xgb.pred = factor(X_test$xgb.pred, c(1,2,3))
  X_test$lgb.pred = factor(X_test$lgb.pred, c(1,2,3))
  names(X_test) = c('x1', 'x2', 'x3', 'x4')
  rownames(X_test)=NULL
  
  mm=model.matrix(Credit_Score~. -1, train)
  glmnet.fit=glmnet(mm, train$Credit_Score, family='multinomial', alpha = alpha_opt , lambda = lambda_opt)
  lr_pred_prob = predict(glmnet.fit, model.matrix(~. -1, X_test), type = 'response')
  pred=argmax(lr_pred_prob)
  pred=factor(pred, levels = c(1,2,3))
  lr.table=table(pred, test$Credit_Score)
  
  res.acc = rep(NA, 4)
  res.acc[1]=accuracy.func(lr.table)[1]
  res.acc[2]=sensitivity.func(lr.table)
  res.acc[3]=specificity.func(lr.table)
  res.acc[4]=accuracy.func(lr.table)[2]
  res.acc[5]=ratio.fun(lr.table)
  
  return(list(res.acc, pred))
}

#################### #################### #################### #################### ####################
#################### ####################      실 행           #################### ####################
#################### #################### #################### #################### ####################

# file_path = "C:/Users/joonmopark/OneDrive/바탕화~1-DESKTOP-6FC9L3H-884275/데이터마이닝"
# img_path = "C:/Users/joonmopark/OneDrive/바탕화~1-DESKTOP-6FC9L3H-884275/데이터마이닝/plot"
# 
# setwd(file_path)

#file_list = c('df_0524.csv')
row_nums=c(8)    #해당 부분에 번호 넣으면 됩니다!!!!!   
file_name='df10000.csv'

for(row_num in row_nums){
  train_opt = data.param[row_num,1]
  test_opt = data.param[row_num,2]
  random_opt = data.param[row_num,3]
  
  num_model=7 + nrow(weight.vec)
  month.acc.mat=month.sen.mat=month.spc.mat=month.fixed_acc.mat=month.ratio.mat=matrix(NA, num_model*8, 100)
  
  X_df=read.csv(file_name)
  
  X_df$Credit_Score=factor(X_df$Credit_Score, levels=c(1, 2, 3))
  X_df$Before_Credit_Score=factor(X_df$Before_Credit_Score, levels=c(1, 2, 3))
  X_df$Occupation = as.factor(X_df$Occupation)
  
  unique_Customer_ID=unique(X_df$Customer_ID)  #1250개
  sampling_ratio=0.3
  nsamples=length(unique_Customer_ID)*sampling_ratio
  start_iteration=1; end_iteration=20
  
  #시작
  start_time=Sys.time()
  
  # setwd(img_path) # plot을 저장할 하위 디렉토리로 경로 변경 
 
  for(iter in start_iteration:end_iteration){
    print(iter)
    sampled_unique_Customer_ID=sample(unique_Customer_ID, nsamples)
    
    test=subset(X_df, (Customer_ID %in% sampled_unique_Customer_ID))
    train=subset(X_df, !(Customer_ID %in% sampled_unique_Customer_ID))
    
    train$Credit_Score=relevel(train$Credit_Score, 1)
    
    test=test%>%mutate(Before_Credit_Score=NA)
    rep_num = 8
    if(train_opt == 'drop'){ train = drop.fun(train) ; rep_num = 7
    } else if(train_opt == "rf_train"){ train = rf.january.func(train, train, train_opt)
    } else if(train_opt == 'mode'){ 
      for(i in 1:nrow(train)){
        if(train$Month[i] == 1){
          train$Before_Credit_Score[i] = 2
        }
      }
    }
    
    if(test_opt=="rf_test"){ test = rf.january.func(train, test, test_opt) 
    } else if(test_opt == 'mode'){
      test = test%>%mutate(Before_Credit_Score=ifelse(Month==1, 2, NA))}
    
    if(random_opt == 'random_o'){ train = random.fun(train, rep_num) }
    
    train$Before_Credit_Score = factor(train$Before_Credit_Score, c(1,2,3))
    test$Before_Credit_Score = factor(test$Before_Credit_Score, c(1,2,3))
    
    test=subset(test, select=-c(X))
    train=subset(train, select=-c(X))
    
    # test=subset(test, select=-c(X, Month))
    # train=subset(train, select=-c(X, Month))
    
    #train = train[1:1000, ]
    
    ##################### Elastic #######################
    print("LR")
    lr.tun.res=lr.tun(train, test)
    
    lr_opt=which.max(lr.tun.res)
    param.opt.mat[iter, 1] = lr_opt
    
    alpha_opt = as.numeric(lr.param[lr_opt, 'alpha.vec'])
    lambda_opt = as.numeric(lr.param[lr_opt, 'lambda.vec'])
    
    # df_2_acc = data.frame(lr.param, lr.tun.res)
    # colnames(df_2_acc)=c("lambda", "alpha", "comb", 'validation')
    # 
    # ggplot(df_2_acc, aes(x=comb, group = 1)) +
    #   geom_line(aes(y =validation, color="red")) +
    #   scale_x_discrete(limits=df_2_acc$comb) + 
    #   labs(title = 'Elastic', y = 'Accuracy', x = 'Hyper Parameter') +
    #   theme(plot.title = element_text(hjust = 0.5)) +
    #   theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))+
    #   theme(legend.position = "none")
    # 
    # model_name='Elastic'
    # save_name= paste(iter,'_',model_name,'.pdf',sep='')
    # ggsave(save_name,width=20,height=15,units = c('cm'))
    
    lr.stack.vec=lr.stack.opt(train, alpha_opt, lambda_opt)
    stack_y = lr.stack.vec[[2]]
    lr.stack.vec = lr.stack.vec[[1]]
    
    train.stack=make.train.stack(train)
    lr.stack.test.vec=lr.optimal.func(train.stack, test, alpha_opt, lambda_opt)
    
    lr.res.vec=lr.optimal.func(train, test, alpha_opt, lambda_opt)
    lr.prob.mat = lr.res.vec[[2]]
    lr.prob.mat =  matrix(lr.prob.mat, length(lr.prob.mat)/3,3)
    lr.res.vec = lr.res.vec[[1]]
    
    acc.mat[iter, 1]=lr.res.vec[1]
    sen.mat[iter, 1]=lr.res.vec[2]
    spc.mat[iter, 1]=lr.res.vec[3]
    fixed_acc.mat[iter, 1]=lr.res.vec[4]
    ratio.mat[iter, 1] = lr.res.vec[5]
    
    y_pred=apply(lr.prob.mat, 1, which.max)
    month.lr.res.mat = month.compare.func(y_pred)
    
    month.acc.mat[1:8, iter]=month.lr.res.mat[,1]
    month.sen.mat[1:8, iter]=month.lr.res.mat[,2]
    month.spc.mat[1:8, iter]=month.lr.res.mat[,3]
    month.fixed_acc.mat[1:8, iter]=month.lr.res.mat[,4]
    month.ratio.mat[1:8,iter] = month.lr.res.mat[,5]
    
    ####################### KNN #####################
    # print("KNN")
    # knn.neighbor.res=knn.tun(train, neighbor.vec)
    # 
    # knn_opt = which.max(knn.neighbor.res)
    # param.opt.mat[iter, 2] = knn_opt
    # 
    # neighbor_opt= as.numeric(neighbor.vec[knn_opt])
    # 
    # knn.res.vec=knn.optimal.func(train, test, neighbor_opt)
    # acc.mat[iter, 2]=knn.res.vec[1]
    # sen.mat[iter, 2]=knn.res.vec[2]
    # spc.mat[iter, 2]=knn.res.vec[3]
    # fixed_acc.mat[iter, 2]=knn.res.vec[4]
    
    mid_time=Sys.time()
    print(mid_time - start_time)
    
    # ########################## RF #########################################
    # print("RandomForest")
    # tree.result = tree.tun(train,test,rf.param =rf.param)
    # 
    # opt_tree = which.max(tree.result)
    # param.opt.mat[iter, 3] = opt_tree
    # 
    # opt_ntree =as.integer(rf.param[opt_tree, 'ntree'])
    # opt_mtry = as.integer(rf.param[opt_tree, 'mtry'])
    # opt_nodesize = as.integer(rf.param[opt_tree, 'nodesize'])
    # 
    # # df_2_acc = data.frame(rf.param, tree.result)
    # # colnames(df_2_acc)=c("ntree", "mtry", "nodesize", 'comb','validation')
    # # 
    # # ggplot(df_2_acc, aes(x=comb, group = 1)) +
    # #   geom_line(aes(y =validation, color="red")) +
    # #   labs(title = 'RandomForest', y = 'Accuracy', x = 'Hyper Parameter') +
    # #   theme(plot.title = element_text(hjust = 0.5)) +
    # #   theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))+
    # #   theme(legend.position = "none")
    # # 
    # # model_name='RandomForest'
    # # save_name= paste(iter,'_',model_name,'.pdf',sep='')
    # # ggsave(save_name,width=20,height=15,units = c('cm'))
    # 
    # tree.stack.vec=tree.stack.opt(train, ntree = opt_ntree, mtry = opt_mtry, nodesize = opt_nodesize) 
    # 
    # tree.stack.test.vec=tree.optimal(train.stack, test, ntree = opt_ntree, mtry = opt_mtry, nodesize = opt_nodesize)
    # 
    # tree.res.vec=tree.optimal(train, test, ntree = opt_ntree, mtry = opt_mtry, nodesize = opt_nodesize)
    # tree.prob.mat = tree.res.vec[[2]]
    # tree.res.vec = tree.res.vec[[1]]
    # 
    # acc.mat[iter, 3]=tree.res.vec[1]
    # sen.mat[iter, 3]=tree.res.vec[2]
    # spc.mat[iter, 3]=tree.res.vec[3]
    # fixed_acc.mat[iter, 3]=tree.res.vec[4]
    # ratio.mat[iter, 3] = tree.res.vec[5]
    # 
    # y_pred=apply(tree.prob.mat,1,which.max)
    # month.tree.res.mat = month.compare.func(y_pred)
    # 
    # month.acc.mat[17:24, iter]=month.tree.res.mat[,1]
    # month.sen.mat[17:24, iter]=month.tree.res.mat[,2]
    # month.spc.mat[17:24, iter]=month.tree.res.mat[,3]
    # month.fixed_acc.mat[17:24, iter]=month.tree.res.mat[,4]
    # month.ratio.mat[17:24,iter] = month.tree.res.mat[,5]
    # 
    # mid_time=Sys.time()
    # print(mid_time - start_time)
    # 
    # ########################## XGBoost ####################################
    # print("XGBoost")
    # xgb.tun.res=xgb.tun(train,test)
    # 
    # xgb_opt=which.max(xgb.tun.res)
    # param.opt.mat[iter, 4] = xgb_opt   #Optimal param 저장장
    # 
    # eta_opt=as.numeric(xgb.param[xgb_opt, 'eta.vec'])
    # max_depth_opt=as.numeric(xgb.param[xgb_opt, 'max_depth.vec'])
    # early.stop.round_opt=as.numeric(xgb.param[xgb_opt, 'early.stop.round.vec'])
    # nrounds_opt=as.numeric(xgb.param[xgb_opt, 'nrounds.vec'])
    # 
    # # df_2_acc = data.frame(xgb.param, xgb.tun.res)
    # # colnames(df_2_acc)=c("eta", "max_depth", "early.stop.round", 'nrounds',
    # #                      'comb','validation')
    # # 
    # # ggplot(df_2_acc, aes(x=comb, group = 1)) +
    # #   geom_line(aes(y =validation, color="red")) +
    # #   scale_x_discrete(limits=df_2_acc$comb) + 
    # #   labs(title = 'XGBoost', y = 'Accuracy', x = 'Hyper Parameter') +
    # #   theme(plot.title = element_text(hjust = 0.5)) +
    # #   theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))+
    # #   theme(legend.position = "none")
    # # 
    # # model_name='XGBoost'
    # # save_name= paste(iter,'_',model_name,'.pdf',sep='')
    # # ggsave(save_name,width=20,height=15,units = c('cm'))
    # 
    # xgb.stack.vec=xgb.stacking.func(train, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # 
    # xgb.stack.test.vec=xgb.optimal.func(train.stack, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # 
    # xgb.res.vec=xgb.optimal.func(train, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # xgb.prob.mat = xgb.res.vec[[2]]
    # xgb.res.vec = xgb.res.vec[[1]]
    # 
    # acc.mat[iter, 4]=xgb.res.vec[1]
    # sen.mat[iter, 4]=xgb.res.vec[2]
    # spc.mat[iter, 4]=xgb.res.vec[3]
    # fixed_acc.mat[iter, 4]=xgb.res.vec[4]
    # ratio.mat[iter, 4] = xgb.res.vec[5]
    # 
    # 
    # y_pred=apply(xgb.prob.mat,1,which.max)
    # 
    # month.xgb.res.mat = month.compare.func(y_pred)
    # 
    # month.acc.mat[25:32, iter]=month.xgb.res.mat[,1]
    # month.sen.mat[25:32, iter]=month.xgb.res.mat[,2]
    # month.spc.mat[25:32, iter]=month.xgb.res.mat[,3]
    # month.fixed_acc.mat[25:32, iter]=month.xgb.res.mat[,4]
    # month.ratio.mat[25:32, iter] = month.xgb.res.mat[,5]
    # 
    # 
    # mid_time=Sys.time()
    # print(mid_time - start_time)
    # 
    # ########################## LGBM ####################################
    # print('LGBM')
    # lgb.tun.res=lgbm.tun(train,test)
    # 
    # lgb_opt=which.max(lgb.tun.res)
    # param.opt.mat[iter, 5] = lgb_opt
    # 
    # eta_opt=as.numeric(lgb.param[lgb_opt, 'eta.vec']) 
    # max_depth_opt=as.numeric(lgb.param[lgb_opt, 'max_depth.vec'])
    # early.stop.round_opt=as.numeric(lgb.param[lgb_opt, 'early.stop.round.vec'])
    # nrounds_opt=as.numeric(lgb.param[lgb_opt, 'nrounds.vec'])
    # 
    # # df_2_acc = data.frame(lgb.param, lgb.tun.res)
    # # colnames(df_2_acc)=c("eta", "max_depth", "early.stop.round", 'nrounds',
    # #                      'comb','validation')
    # # 
    # # ggplot(df_2_acc, aes(x=comb, group = 1)) +
    # #   geom_line(aes(y =validation, color="red")) +
    # #   scale_x_discrete(limits=df_2_acc$comb) + 
    # #   labs(title = 'LGBM', y = 'Accuracy', x = 'Hyper Parameter') +
    # #   theme(plot.title = element_text(hjust = 0.5)) +
    # #   theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))+
    # #   theme(legend.position = "none")
    # # 
    # # model_name='LGBM'
    # # save_name= paste(iter,'_',model_name,'.pdf',sep='')
    # # ggsave(save_name,width=20,height=15,units = c('cm'))
    # 
    # lgb.stack.vec = lgbm.stacking.func(train, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # lgb.stack.test.vec = lgbm.optimal.func(train.stack, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # 
    # lgb.res.vec = lgbm.optimal.func(train, test, eta=eta_opt, max_depth=max_depth_opt, early_stopping_rounds=early.stop.round_opt, nrounds=nrounds_opt)
    # 
    # lgb.prob.mat=lgb.res.vec[[2]]
    # lgb.res.vec = lgb.res.vec[[1]]
    # 
    # acc.mat[iter, 5]=lgb.res.vec[1]
    # sen.mat[iter, 5]=lgb.res.vec[2]
    # spc.mat[iter, 5]=lgb.res.vec[3]
    # fixed_acc.mat[iter, 5]=lgb.res.vec[4]
    # ratio.mat[iter, 5] = lgb.res.vec[5]
    # 
    # y_pred=apply(lgb.prob.mat,1,which.max)
    # month.lgb.res.mat = month.compare.func(y_pred)
    # 
    # month.acc.mat[33:40, iter]=month.lgb.res.mat[,1]
    # month.sen.mat[33:40, iter]=month.lgb.res.mat[,2]
    # month.spc.mat[33:40, iter]=month.lgb.res.mat[,3]
    # month.fixed_acc.mat[33:40, iter] = month.lgb.res.mat[,4]
    # month.ratio.mat[33:40, iter] = month.lgb.res.mat[,5]
    # 
    # mid_time=Sys.time()
    # print(mid_time - start_time)
    # 
    # ########################## NB ####################################
    # # print('naiveBayes')
    # # nb.res.vec = nb.optimal.func(train, test)
    # # 
    # # acc.mat[iter, 6] = nb.res.vec[1]
    # # sen.mat[iter, 6] = nb.res.vec[2]
    # # spc.mat[iter, 6] = nb.res.vec[3]
    # # fixed_acc.mat[iter, 6]=nb.res.vec[4]
    # # 
    # 
    # ####################### Stacking ###############################
    # print('stacking')
    # rf.stack.vec = rf.stacking.fun(x1 = lr.stack.vec, x2 = tree.stack.vec, x3 = xgb.stack.vec, x4 = lgb.stack.vec, Credit_Score = stack_y,
    #                          y1 = lr.stack.test.vec[[2]], y2 = tree.stack.test.vec[[2]], y3 = xgb.stack.test.vec[[2]], y4 = lgb.stack.test.vec[[2]], test = test)
    # 
    # acc.mat[iter, 6]=rf.stack.vec[[1]][1]
    # sen.mat[iter, 6]=rf.stack.vec[[1]][2]
    # spc.mat[iter, 6]=rf.stack.vec[[1]][3]
    # fixed_acc.mat[iter, 6]=rf.stack.vec[[1]][4]
    # ratio.mat[iter, 6] = rf.stack.vec[[1]][5]
    # 
    # month.rf.stack.mat = month.compare.func(rf.stack.vec[[2]])
    # 
    # month.acc.mat[41:48, iter]=month.rf.stack.mat[,1]
    # month.sen.mat[41:48, iter]=month.rf.stack.mat[,2]
    # month.spc.mat[41:48, iter]=month.rf.stack.mat[,3]
    # month.fixed_acc.mat[41:48, iter]=month.rf.stack.mat[,4]
    # month.ratio.mat[41:48, iter] = month.rf.stack.mat[,5]
    # 
    # 
    # lr.stack.vec = lr.stacking.fun(x1 = lr.stack.vec, x2 = tree.stack.vec, x3 = xgb.stack.vec, x4 = lgb.stack.vec, Credit_Score = stack_y,
    #                                y1 = lr.stack.test.vec[[2]], y2 = tree.stack.test.vec[[2]], y3 = xgb.stack.test.vec[[2]], y4 = lgb.stack.test.vec[[2]], test = test)
    # 
    # acc.mat[iter, 7]=lr.stack.vec[[1]][1]
    # sen.mat[iter, 7]=lr.stack.vec[[1]][2]
    # spc.mat[iter, 7]=lr.stack.vec[[1]][3]
    # fixed_acc.mat[iter, 7]=lr.stack.vec[[1]][4]
    # ratio.mat[iter, 7] = lr.stack.vec[[1]][5]
    # 
    # month.lr.stack.mat = month.compare.func(lr.stack.vec[[2]])
    # 
    # month.acc.mat[49:56, iter]=month.lr.stack.mat[,1]
    # month.sen.mat[49:56, iter]=month.lr.stack.mat[,2]
    # month.spc.mat[49:56, iter]=month.lr.stack.mat[,3]
    # month.fixed_acc.mat[49:56, iter]=month.lr.stack.mat[,4]
    # month.ratio.mat[49:56, iter] = month.lr.stack.mat[,5]
    # 
    # 
    # ####################### Ensemble ###############################
    # print('ensemble')
    # ensemble.res.vec = NULL;
    # for(ind in 1:nrow(weight.vec)){
    #   ensemble.res.vec = rbind(ensemble.res.vec, ensemble.fun(weight.vec[ind,] ,m1 =lr.prob.mat, m2 = tree.prob.mat , m3 = xgb.prob.mat,m4 = lgb.prob.mat,Y_TEST =test$Credit_Score)[[1]])
    # }
    # 
    # month.ensemble.res.vec = NULL
    # for(ind in 1:nrow(weight.vec)){
    #   y_pred=ensemble.fun(weight.vec[ind,], m1 =lr.prob.mat, m2 = tree.prob.mat , m3 = xgb.prob.mat,m4 = lgb.prob.mat,Y_TEST =test$Credit_Score)[[2]]
    #   month.ensemble.res.vec = rbind(month.ensemble.res.vec, month.compare.func(y_pred))
    # }
    # 
    # for(pos in 8:num_model){
    #   acc.mat[iter, pos] = ensemble.res.vec[pos-7,1]
    #   sen.mat[iter, pos] = ensemble.res.vec[pos-7,2]
    #   spc.mat[iter, pos] = ensemble.res.vec[pos-7,3]
    #   fixed_acc.mat[iter, pos]=ensemble.res.vec[pos-7,4]
    #   ratio.mat[iter, pos] = ensemble.res.vec[pos-7,5]
    #   
    #   month.acc.mat[(8*(pos-1)+1) : (8*(pos)), iter] = month.ensemble.res.vec[(8*(pos-8)+1) : (8*(pos-7)),1]
    #   month.sen.mat[(8*(pos-1)+1) : (8*(pos)), iter] = month.ensemble.res.vec[(8*(pos-8)+1) : (8*(pos-7)),2]
    #   month.spc.mat[(8*(pos-1)+1) : (8*(pos)), iter] = month.ensemble.res.vec[(8*(pos-8)+1) : (8*(pos-7)),3]
    #   month.fixed_acc.mat[(8*(pos-1)+1) : (8*(pos)), iter] = month.ensemble.res.vec[(8*(pos-8)+1) : (8*(pos-7)),4]
    #   month.ratio.mat[(8*(pos-1)+1) : (8*(pos)), iter] = month.ensemble.res.vec[(8*(pos-8)+1) : (8*(pos-7)),5]
      
      
      # 기존에 7개 모형이 존재하므로 -7
    # }
  }
  
  end_time=Sys.time()
  amount_time=end_time-start_time; amount_time
  
  #저장
  ############# model2 #############
  acc.mat = acc.mat[start_iteration:end_iteration, ]
  sen.mat = sen.mat[start_iteration:end_iteration, ]
  spc.mat = spc.mat[start_iteration:end_iteration, ]
  fixed_acc.mat = fixed_acc.mat[start_iteration:end_iteration, ]
  ratio.mat = ratio.mat[start_iteration:end_iteration, ]
  
  save.mat=cbind(c(start_iteration:end_iteration), acc.mat, sen.mat, spc.mat,fixed_acc.mat, ratio.mat )
  
  model_list =c('LR','KNN','RF','XGB','LGB','RF.STACK','LR.STACK')
  
  for (i in 1:nrow(weight.vec)){
    ens_name = paste(weight.vec[i,1] , weight.vec[i,2],weight.vec[i,3],weight.vec[i,4], sep='/')
    model_list =append(model_list, ens_name, length(model_list))
  }
  metric_list = c('ACC' , 'SEN','SPC', "CUSTOM_ACC", 'RATIO')
  
  col_name = c('iteration') 
  for (metric in metric_list){
    for(model in model_list){
      cname=  paste(metric , model, sep='_')
      col_name = rbind(col_name ,cname )
    }
  }
  
  colnames(save.mat) = col_name
  save_name = paste('model2_',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(save.mat, file=save_name)
  
  acc.mean.mat = apply(save.mat, 2, mean)
  mean_name = paste('model2_mean_',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(acc.mean.mat, file = mean_name)
  
  param_name = paste('model2_param_',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(param.opt.mat, file = param_name)
  
  
  
  ############# model3 #############
  month.acc.mat = month.acc.mat[, start_iteration:end_iteration]
  month.sen.mat = month.sen.mat[, start_iteration:end_iteration]
  month.spc.mat = month.spc.mat[, start_iteration:end_iteration]
  month.fixed_acc.mat = month.fixed_acc.mat[, start_iteration:end_iteration]
  month.ratio.mat = month.ratio.mat[, start_iteration:end_iteration]
  
  col_name = 0:end_iteration 
  
  model_list =c('LR','KNN','RF','XGB','LGB','RF.STACK','LR.STACK')
  
  for (i in 1:nrow(weight.vec)){
    ens_name = paste(weight.vec[i,1] , weight.vec[i,2],weight.vec[i,3],weight.vec[i,4], sep='/')
    model_list =append(model_list, ens_name, length(model_list))
  }
  metric_list = 1:8
  
  row_name = NULL
  for(model in model_list){
    for (metric in metric_list){
      cname=  paste('month', metric , model, sep='_')
      row_name = rbind(row_name ,cname )  
    }
  }
  month.acc.mat = cbind(as.vector(row_name),month.acc.mat)
  month.spc.mat = cbind(as.vector(row_name),month.spc.mat)
  month.sen.mat = cbind(as.vector(row_name),month.sen.mat)
  month.fixed_acc.mat = cbind(as.vector(row_name),month.fixed_acc.mat)
  month.ratio.mat = cbind(as.vector(row_name),month.ratio.mat)
  
  colnames(month.acc.mat) = col_name
  save_name = paste('model3_acc',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(month.acc.mat, file=save_name)
  
  colnames(month.spc.mat) = col_name
  save_name = paste('model3_spc',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(month.spc.mat, file=save_name)
  
  colnames(month.sen.mat) = col_name
  save_name = paste('model3_sen',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(month.sen.mat, file=save_name)
  
  colnames(month.fixed_acc.mat) = col_name
  save_name = paste('model3_fixed_acc',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(month.fixed_acc.mat, file=save_name)
  
  colnames(month.ratio.mat) = col_name
  save_name = paste('model3_ratio',row_num,'_',start_iteration,'_',end_iteration,'_',file_name,sep='')
  write.csv(month.ratio.mat, file=save_name)
}

