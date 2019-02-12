#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:58:20 2018

"""
#import sys
#sys.path.append("/Quartic")
from QuarticApi import QuarticApp as Qa
import QuarticApi
from sklearn import naive_bayes,model_selection,linear_model,neighbors
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,f1_score
from sklearn.svm import LinearSVC
import pandas as pd
Obj=Qa()
Obj.read_data()
train_data=Obj.train_data
train_data.columns
train_data.info()
test_data=Obj.test_data

#looking for imputed and nonimputed features
Obj.ValidateColumns()

#bayes classifier
naive_cl=naive_bayes.GaussianNB()
x=train_data[Obj.nonim_columns]
x=x.iloc[:,:43]
y=train_data["target"]
naive_cl.fit(x,y)
print(naive_cl.class_prior_)  #[0.96356376 0.03643624] clearly imbalanced
print(naive_cl.sigma_)
print(naive_cl.theta_)  #liklihood

len(train_data[train_data.target==0]) #574284
len(train_data[train_data.target==1]) #21716

#test,train split and confusion matrix , roc curve
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.33, random_state=42)
naive_cl.fit(x_train,y_train)
 
y_predict=naive_cl.predict(x_test)
average_precision_score(y_test,y_predict)
tn, fp, fn,tp=confusion_matrix(y_test, y_predict).ravel()
QuarticApi.roc_binary(y_test,y_predict)
precision_recall_curve(y_test,y_predict)
f1_score(y_test,y_predict)

#knn classifier  better classifier till
knn=neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_knn=knn.predict(x_test)
average_precision_score(y_knn,y_test)
QuarticApi.roc_binary(y_test,y_knn)
confusion_matrix(y_test, y_knn)


#svc  resampling required 
sv_clf = LinearSVC(random_state=0, tol=1e-5)
sv_clf.fit(x_train,y_train)
sv_clf.decision_function(x_train)
y_predict=sv_clf.predict(x_test)

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







#test script
test_data['target']=None
test_data['target']=naive_cl.predict(test_data[nonim_columns])




