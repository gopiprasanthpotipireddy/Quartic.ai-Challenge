#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:58:20 2018

"""
#import sys
#sys.path.append("/Quartic")
from QuarticApi import read_data
from sklearn import naive_bayes,model_selection,linear_model
train_data=read_data()
train_data.columns
train_data.info()
features=list(train_data.columns)
im_features=[]
nonim_columns=[]

def ValidateColumns():    #checking for imputed and non imputed columns
    
    for i in range(1,len(train_data.columns)):
        if train_data.iloc[:,i].isna().sum() == 0:
            nonim_columns.append(features[i])
        else:
            im_features.append(features[i])

ValidateColumns()

#bayes classifier
naive_cl=naive_bayes.GaussianNB()
x=train_data[nonim_columns]
y=train_data["target"]
naive_cl.fit(x,y)
#print(naive_cl.class_prior_)  #[0.96356376 0.03643624] clearly imbalanced
#print(naive_cl.sigma_)
#print(naive_cl.theta_)  #liklihood

len(train_data[train_data.target==0]) #574284
len(train_data[train_data.target==1]) #21716

#crossvalidation

res = model_selection.cross_validate(naive_cl, x, y, cv=20)
res.get('test_score').mean()
res.get('train_score').mean()
naive_cl.predict(x[1:4])
x[1:2]
#logistic regression
lr=linear_model.LogisticRegression()
lr_res=model_selection.cross_validate(lr,x,y,cv=10)
lr.fit(x,y)
#list(lr.predict(x))==y

sample=train_data[1:10000]
naive_cl.fit(sample[nonim_columns],sample["target"])
pred=naive_cl.predict(train_data.loc[10000:,nonim_columns])
t = model_selection.cross_validate(naive_cl, sample[nonim_columns], sample["target"], cv=20)
list(train_data.loc[10000:,"target"])
list(pred)


