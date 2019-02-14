# -*- coding: utf-8 -*-

from QuarticApi import QuarticApp
import QuarticApi as Qa
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.svm import LinearSVC
from sklearn import model_selection
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,f1_score,precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import numpy as np
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

SMOTE_Sampler=SMOTE()
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5, random_state=42)
x_resampled, y_resampled = SMOTE_Sampler.fit_resample(x_train, y_train)


svc=LinearSVC()

param_grid={
        'C': np.linspace(1,3, 10)    
             }
cl=GridSearchCV(svc,param_grid,cv=5,verbose=5,n_jobs=3)

cl.fit(x_resampled,y_resampled)
cl.best_params_
cl.best_score_
cl.cv_results_
#cl.fit(x_train,y_train)

y_pred=cl.predict(x_test)
confusion_matrix(y_test,y_pred)
average_precision_score(y_test,y_pred)
precision_recall_curve(y_test,y_pred)
f1_score(y_test,y_pred)

roc_binary(y_test,y_pred)
