#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:06:17 2018

@author: gopiprasanth
"""
import os
import pandas as pd
from sklearn.metrics import average_precision_score,confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class QuarticApp:
    def __init__(self):
        self.train_data = None
        self.test_data= None
        return
    
    def read_data(self):
        file_location="/home/gopiprasanth/Desktop/Algorithmica/Quartic.ai/ds_data_big/ds_data"
        self.train_data=pd.read_csv(os.path.join(file_location,"data_train.csv"))
        self.test_data=pd.read_csv(os.path.join(file_location,"data_test.csv"))
        #train_data.info()
        #test_data=pd.read_csv(os.path.join(file_location,"data_test.csv"))
        #test_data.info()
        return

    def build_model(self,model_type):
        #model
        return
    
    def ValidateColumns(self):    #checking for imputed and non imputed columns
        self.features=list(self.train_data.columns)
        self.im_features=[]
        self.nonim_columns=[]
        for i in range(1,len(self.train_data.columns)):
            if self.train_data.iloc[:,i].isna().sum() == 0:
                self.nonim_columns.append(self.features[i])
            else:
                self.im_features.append(self.features[i])
                    
        return 

def roc_binary(y_test,y_predict):
    
    fpr, tpr,_ = roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()    
    
    
    

    
    


    