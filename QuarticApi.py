#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:06:17 2018

@author: gopiprasanth
"""
import os
import pandas as pd


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


    
    
    

    
    


    