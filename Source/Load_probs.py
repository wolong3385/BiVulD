# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:53:04 2018


"""
import pandas as pd
import pickle

working_dir = 'D:\\data_path\\'

def LoadSavedData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

probs = LoadSavedData(working_dir + 'probs.pickle')

predicted_classes = []

for item in probs:
    if item[0] > 0.5:
        predicted_classes.append(1)
    else:
        predicted_classes.append(0)
        
def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)