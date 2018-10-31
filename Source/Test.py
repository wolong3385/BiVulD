# -*- coding: utf-8 -*-
"""



"""

from random import randrange
import os
import csv
import pickle
import numpy as np
import pandas as pd

from keras.layers import Input
from keras.models import load_model, Model

from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


LOSS_FUNCTION = 'binary_crossentropy'
#OPTIMIZER = 'adamax'
OPTIMIZER = 'adamax'

MAX_LEN = 500
EMBEDDING_DIM = 100
BATCH_SIZE = 16

working_dir = 'N:\\data_path\\'
#working_dir = 'D:\\Phd\\Backup\\2018-08-23_char_level_LSTM\\Binary-level-repre\\assembly code\\assemblely\\'

model_saved_path = 'N:\\data_path\\models\\'
#model_saved_path = working_dir 
# model name: navigate to model file
model_name = 'BiLSTM_binary_attention_07_0.744_0.445203.h5'

model = load_model(model_saved_path + model_name)

model.compile(loss= LOSS_FUNCTION,
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

print ("The model has been loaded: ")
print (model.summary())

def LoadSavedData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

test_set_x = LoadSavedData(working_dir + 'test_set_x.pickle')
test_set_y = LoadSavedData(working_dir + 'test_set_y.pickle')

print (test_set_x.shape)
#
#def JoinSubLists(list_to_join):
#    new_list = []
#    
#    for sub_list_token in list_to_join:
#        new_line = ','.join(sub_list_token)
#        new_list.append(new_line)
#    return new_list
#
#test_list = JoinSubLists(test_set_x)
#
#tokenizer = LoadSavedData(working_dir + 'assembly_tokenizer.pickle')
#test_sequences = tokenizer.texts_to_sequences(test_list)
#
#test_sequences_pad = pad_sequences(test_sequences, maxlen = MAX_LEN, padding ='post')

print ("max_len ", MAX_LEN)
print('Pad sequences (samples x time)')

probs = model.predict(test_set_x, batch_size = BATCH_SIZE, verbose=1)

with open(working_dir + 'probs.pickle', 'wb') as handle:
    #pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(probs, handle, protocol=2)

ListToCSV(probs.tolist(), 'prob_assembly.csv')

predicted_classes = []

for item in probs:
    if item[0] > 0.5:
        predicted_classes.append(1)
    else:
        predicted_classes.append(0)

ListToCSV(predicted_classes, 'classes_assembly.csv')

test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))

test_set_y = np.asarray(test_set_y)

print ("LSTM classification result: ")

target_names = ["Non-vulnerable","Vulnerable"] #non-vulnerable->0, vulnerable->1
print (confusion_matrix(test_set_y, predicted_classes, labels=[0,1]))   
print ("\r\n")
print ("\r\n")
print (classification_report(test_set_y, predicted_classes, target_names=target_names))