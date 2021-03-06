#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:55:36 2018

@author: gopiprasanth
"""
from QuarticApi import QuarticApp
import QuarticApi as Qa
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,f1_score,precision_recall_fscore_support

import os
import pandas as pd
Obj=QuarticApp()
Obj.read_data()
train_data=Obj.train_data
train_data.columns
train_data.info()
test_data=Obj.test_data
#looking for imputed and nonimputed features
Obj.ValidateColumns()
"""
for i in range(1,len(Obj.train_data.columns)):
            if Obj.train_data.iloc[:,i].isnull().sum() == 0:
                Obj.nonim_columns.append(Obj.features[i])
            else:
                Obj.im_features.append(Obj.features[i])
    
"""
Obj.nonim_columns.remove('target')
x=train_data[Obj.nonim_columns]
y=train_data["target"]


#Random Oversampling
ros = RandomOverSampler(random_state=0)

x_resampled, y_resampled = ros.fit_resample(x, y)

cl=LinearSVC()
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_resampled, y_resampled, test_size=0.5, random_state=42)
cl.fit(x_train,y_train)

y_pred=cl.predict(x_test)
confusion_matrix(y_test,y_pred)
average_precision_score(y_test,y_pred)
precision_recall_curve(y_test,y_pred)
f1_score(y_test,y_pred)

roc_binary(y_test,y_pred)
#high roc and high f1 score and the classifier is doing a good job

#results feeding entire data
cl.fit(x_resampled,y_resampled)
Obj.nonim_columns.remove('target')
test_sample=test_data[Obj.nonim_columns]

test_data["target"]=None
test_data["target"]=cl.predict(test_sample)

file_location=""
test_data.loc[:,['id','target']].to_csv("results.csv",index=False)
precision_recall_fscore_support(y_test,y_pred)
