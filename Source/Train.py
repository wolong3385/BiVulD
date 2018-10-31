# -*- coding: utf-8 -*-
"""

"""
import os
import pandas as pd
import csv
import pickle
import numpy as np
import datetime

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger

from BiLSTM_model_with_Attention_GPU import BiLSTM_network

#from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, merge, concatenate

MAX_LEN = 500
EMBEDDING_DIM = 100
EPOCHS = 1
BATCH_SIZE = 32
PATIENCE = 40

#working_dir = '/home/huawei1_lingu/guanjun/Assembly/'
working_dir = 'N:\\data_path\\'
model_saved_path = working_dir + 'models'
log_path = working_dir + 'logs'

# 1. Load data.

def getData(file_path):
    
    # The encoding has to be 'latin1' to make sure the string to float convertion to be smooth
    df = pd.read_csv(file_path, sep=',', encoding='latin1', low_memory=False, header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False) 
    df_list = df.values.tolist()
    
    temp = []
    for i in df_list:
        # Get rid of 'NaN' values.
        i = [x for x in i if str(x) != 'nan']
        i = [str(x) for x in i]
        new_i = []
        for item in i:
            sub_temp = item.split()
            for sub_item in sub_temp:
                new_i.append(sub_item)
        temp.append(new_i)
    
    return temp





def getID(file_path):
    
    # The encoding has to be 'latin1' to make sure the string to float convertion to be smooth
    df = pd.read_csv(file_path, sep=',', encoding='latin1', low_memory=False, header=None) 
    df_list = df.values.tolist()
    
    return df_list

Assembly_bad_list = getData(working_dir + 'binary_bad.csv')
Assembly_bad_label_list = [1] * len(Assembly_bad_list)
Assembly_bad_id_list = getID(working_dir + 'binary_bad_id.csv')

Assembly_good_list = getData(working_dir + 'binary_good.csv')
Assembly_good_label_list = [0] * len(Assembly_good_list)
Assembly_good_id_list = getID(working_dir + 'binary_good_id.csv')

total_list = Assembly_bad_list + Assembly_good_list
total_list_label = Assembly_bad_label_list + Assembly_good_label_list
total_list_id = Assembly_bad_id_list + Assembly_good_id_list
#--------------------------------------------------------#
# 2. Tokenization: convert the loaded text to tokens.

new_total_token_list = []

for sub_list_token in total_list:
    new_line = ','.join(sub_list_token)
    new_total_token_list.append(new_line)

tokenizer = Tokenizer(num_words=None, filters=',', lower=False, char_level=False)
tokenizer.fit_on_texts(new_total_token_list)

# Save the tokenizer.
with open(working_dir + 'binary_tokenizer.pickle', 'wb') as handle:
    #pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(tokenizer, handle, protocol=2)
    
# ----------------------------------------------------- #
# 3. Train a Vocabulary with Word2Vec -- using the function provided by gensim

#w2vModel = Word2Vec(train_token_list, workers = 12, size=100) # With default settings, the embedding dimension is 100 and using, (sg=0), CBOW is used.  
w2vModel = Word2Vec(total_list, workers = 12)

print ("----------------------------------------")
print ("The trained word2vec model: ")
print (w2vModel)

w2vModel.wv.save_word2vec_format(working_dir + "binary_model_CBOW.txt", binary=False)

w2v_model_path = working_dir + 'binary_model_CBOW.txt'
w2v_model = open(w2v_model_path)

# 4. Use the trained tokenizer to tokenize the sequence.
#-------------------------------------------------------------------
total_sequences = tokenizer.texts_to_sequences(new_total_token_list)
word_index = tokenizer.word_index
print ('Found %s unique tokens.' % len(word_index))

print ("The length of tokenized sequence: " + str(len(total_sequences)))

#------------------------------------#
# 5. Do the paddings.
#--------------------------------------------------------

print ("max_len ", MAX_LEN)
print('Pad sequences (samples x time)')

total_sequences_pad = pad_sequences(total_sequences, maxlen = MAX_LEN, padding ='post')

print ("The shape after paddings: ")
print (total_sequences_pad.shape)

train_set_x, validation_set_x, train_set_y, validation_set_y, train_set_id, validation_set_id = train_test_split(total_sequences_pad, total_list_label, total_list_id, test_size=0.5, random_state=42)

test_set_x, validation_set_x, test_set_y, validation_set_y, test_set_id, validation_set_id = train_test_split(validation_set_x, validation_set_y, validation_set_id, test_size=0.5, random_state=42)






print ("Training set: ")

print (train_set_x)

print ("Validation set: ")

print (validation_set_x)

print ("Test set: ")

print (test_set_x)

# Save the test data sets.
#--------------------------------------------------------------
with open(working_dir + 'test_set_x.pickle', 'wb') as handle:
    #pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set_x, handle, protocol=2)
    
with open(working_dir + 'test_set_y.pickle', 'wb') as handle:
    #pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set_y, handle, protocol=2)

train_set_y = np.asarray(train_set_y)
validation_set_y = np.asarray(validation_set_y)
test_set_y = np.asarray(test_set_y)

print (len(train_set_x), len(train_set_y), len(train_set_id), len(validation_set_x),  len(validation_set_y), len(validation_set_id))

print (train_set_x.shape, train_set_y.shape, validation_set_x.shape, validation_set_y.shape, test_set_x.shape, test_set_y.shape, )

print (np.count_nonzero(train_set_y), np.count_nonzero(validation_set_y), np.count_nonzero(test_set_y))

# -----------------------------------
# 6. Preparing the Embedding layer

embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

# Use the loaded model
for line in w2v_model:
   if not line.isspace():
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
w2v_model.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
   embedding_vector = embeddings_index.get(word)
   if embedding_vector is not None:
       # words not found in embedding index will be all-zeros.
       embedding_matrix[i] = embedding_vector
       
def train(train_set_x, train_set_y, validation_set_x, validation_set_y, saved_model_name):
    
    model = BiLSTM_network(MAX_LEN, EMBEDDING_DIM, word_index, embedding_matrix, True)
    
    callbacks_list = [
       ModelCheckpoint(filepath = model_saved_path + os.sep + saved_model_name +'_{epoch:02d}_{val_acc:.3f}_{val_loss:3f}.h5', monitor='val_loss', verbose=2, save_best_only=True, period=1),
       EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=2, mode="min"),
		TensorBoard(log_dir=log_path, batch_size = BATCH_SIZE,  write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
       CSVLogger(log_path + os.sep + saved_model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')]
    
    model.fit(train_set_x, train_set_y,
         epochs=EPOCHS,
         batch_size=BATCH_SIZE,
		   shuffle = False, # The data has already been shuffle before, so it is unnessary to shuffle it again. (And also, we need to correspond the ids to the features of the samples.)
         #validation_split=0.5,
         validation_data = (validation_set_x, validation_set_y), # Validation data is not used for training (or development of the model)
         callbacks=callbacks_list, # Get the best weights of the model and stop the first raound training.
         verbose=2)
    
    model.summary()
    
    return model

if __name__ == '__main__':

    train(train_set_x, train_set_y, validation_set_x, validation_set_y, 'BiLSTM_binary_attention')