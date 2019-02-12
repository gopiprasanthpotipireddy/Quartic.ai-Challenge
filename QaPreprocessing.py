#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 00:26:52 2018

@author: gopiprasanth
"""
from QuarticApi import QuarticApp as Qa
 

def ValidateColumns(train_data):    #checking for imputed and non imputed columns
    features=list(train_data.columns)
    im_features=[]
    nonim_columns=[]
    for i in range(1,len(train_data.columns)):
        if train_data.iloc[:,i].isna().sum() == 0:
            nonim_columns.append(features[i])
        else:
            im_features.append(features[i])