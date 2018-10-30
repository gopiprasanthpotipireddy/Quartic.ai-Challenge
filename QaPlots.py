#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:38:44 2018

@author: gopiprasanth
"""

#plotting graphs
from QuarticApi import * 
from sklearn.metrics import average_precision_score,confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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