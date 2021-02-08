#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
#from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.models import Sequential,Model
from keras.layers import Dense,Attention,GlobalMaxPooling1D
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, Conv1D,MaxPooling1D,Embedding, LSTM, Input, Flatten, Dense,Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, Activation, Concatenate
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten,Reshape,MaxPool2D
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import re


# In[9]:


# load the dataset
df= pd.read_csv('Constraint_English_Train.csv')
df_val= pd.read_csv('Constraint_English_Val.csv')
df_test= pd.read_csv('Constraint_English_Test.csv')


# In[10]:


df_all = pd.concat([df, df_val])
#df_ALL=pd.concat([df_all,df_test])

X_train=df_all['tweet']
y_train=df_all['label']

X_test= df_test['tweet']
y_test=df_test['label']
#print(df_all.shape)


# In[11]:


# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
tokenizer.fit_on_texts(X_test)
X_test = tokenizer.texts_to_sequences(X_test)


# In[12]:


# Pad_sequence
from keras.preprocessing.sequence import pad_sequences
input_length=1420
X_train = pad_sequences(X_train, padding='post', maxlen=input_length)
X_test = pad_sequences(X_test, padding='post', maxlen=input_length)


# In[13]:


# encoding label
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test= encoder.fit_transform(y_test)


# In[17]:


#word embedding
emb_dim=300
vocab=len(tokenizer.word_index)+1
#print(vocab)

emb_mat= np.zeros((vocab,emb_dim))

with open('glove.6B.300d.txt') as f:
    for line in f:
        word,*emb = line.split()
        if word in tokenizer.word_index:
            ind=tokenizer.word_index[word]
            emb_mat[ind]=np.array(emb,dtype="float32")[:emb_dim]


def model_cnn(emb_mat):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(input_length,))
    x = Embedding(vocab, 300, weights=[emb_mat])(inp)
    x = Reshape((input_length,300, 1))(x)
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], 300), padding='same',activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(input_length - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool) 
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model=model_cnn(emb_mat)
history=model.fit(X_train,y_train,epochs=10,batch_size=64, verbose=2,shuffle=True)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred = np.round(y_pred,0)
#prediction=model.predict(X_test)

scores = model.evaluate(X_test,y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

print('Classification Report:')
print(classification_report(y_test, y_pred, labels=[1,0], digits=4))



# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




