#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:06:17 2018

@author: gopiprasanth
"""
import os
import pandas as pd
def read_data():
    file_location="/home/gopiprasanth/Desktop/Algorithmica/Quartic.ai/ds_data_big/ds_data"
    train_data=pd.read_csv(os.path.join(file_location,"data_train.csv"))

    #train_data.info()
    #test_data=pd.read_csv(os.path.join(file_location,"data_test.csv"))
    #test_data.info()
    return train_data

def build_model(model_type):
    #model
    return


    
    
    

    
    


    