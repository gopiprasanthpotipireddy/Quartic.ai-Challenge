#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:58:20 2018

"""
#import sys
#sys.path.append("/Quartic")
from QuarticApi import QuarticApp as Qa
import QaPLots as plots,QaPreprocessing as prep
from sklearn import naive_bayes,model_selection,linear_model,neighbors
from sklearn.metrics import average_precision_score,confusion_matrix, roc_curve, auc
Obj=Qa()
Obj.read_data()
train_data=Obj.train_data
train_data.columns
train_data.info()
test_data=Obj.test_data
features=list(train_data.columns)
im_features=[]
nonim_columns=[]


prep.ValidateColumns()

    

#bayes classifier
naive_cl=naive_bayes.GaussianNB()
x=train_data[nonim_columns]
y=train_data["target"]
naive_cl.fit(x,y)
print(naive_cl.class_prior_)  #[0.96356376 0.03643624] clearly imbalanced
#print(naive_cl.sigma_)
#print(naive_cl.theta_)  #liklihood

len(train_data[train_data.target==0]) #574284
len(train_data[train_data.target==1]) #21716

#test,train split and confusion matrix , roc curve
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.33, random_state=42)
naive_cl.fit(x_train,y_train)
 
y_predict=naive_cl.predict(x_test)
average_precision_score(y_test,y_predict)
confusion_matrix(y_test, y_predict)
plots.roc_binary()

#knn classifier
knn=neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_knn=knn.predict(x_test)
average_precision_score(y_knn,y_test)

#crossvalidation cv score for naive classifier =1
res = model_selection.cross_validate(naive_cl, x, y, cv=20)
res.get('test_score').mean()
res.get('train_score').mean()
naive_cl.predict(x[1:4])
x[1:2]

#logistic regression making test score =1
lr=linear_model.LogisticRegression()
lr_res=model_selection.cross_validate(lr,x,y,cv=10)
lr.fit(x,y)
list(lr.predict(x))==y


#classifying a sample of dataset
sample=train_data[1:10000]
naive_cl.fit(sample[nonim_columns],sample["target"])
pred=naive_cl.predict(train_data.loc[10000:,nonim_columns])
t = model_selection.cross_validate(naive_cl, sample[nonim_columns], sample["target"], cv=20)
list(train_data.loc[10000:,"target"])
list(pred)



#test script
test_data['target']=None
test_data['target']=naive_cl.predict(test_data[nonim_columns])




