# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:12:23 2019

@author: lisha
"""
import tensorflow as tf
import pandas as pd
import numpy as np


from FFNN_class import *
from RNN_class import *
from CNN1d_class import *
from CNN2d_class import *
from lstm_class import *

X_train = pd.read_csv("x_train.csv",index_col=0)
X_valid = pd.read_csv("x_test.csv",index_col=0)
y_valid = pd.read_csv("y_test.csv",index_col=0)
y_train = pd.read_csv("y_train.csv",index_col=0)


# In[3]:


X_train.shape


# In[4]:


train_data = pd.concat([X_train, y_train], axis=1)
train_data.head()


# In[5]:


test_data = pd.concat([X_valid, y_valid], axis=1)
test_data.head()


# In[6]:


from sklearn.utils import shuffle
train_data = shuffle(train_data) 
test_data = shuffle(test_data)


# In[7]:


X_train = train_data.drop(columns='label')
y_train = train_data['label']
X_valid = test_data.drop(columns='label')
y_valid = test_data['label']


# In[8]:


y_valid = pd.DataFrame(y_valid)
y_train = pd.DataFrame(y_train)


# In[9]:


y_train = y_train.replace(-1, 2)
y_valid = y_valid.replace(-1, 2).values
y_valid = np.asarray([j for i in y_valid for j in i])



# In[363]:


FFNN(X_train,y_train,X_valid,y_valid)
RNN(X_train,y_train,X_valid,y_valid)
CNN_1d(X_train,y_train,X_valid,y_valid)
CNN_2d(X_train,y_train,X_valid,y_valid)
LSTM(X_train,y_train,X_valid,y_valid)