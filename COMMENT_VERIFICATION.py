#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np


# In[49]:


_TITLE_ = 'title'
_COMMENT_ = 'comment'
_TITLE_AND_COMMENT_ = 'title_comment'
_VERIF_STAT_ = 'verification_status'
_ID_ = 'id'

_CONST_ = 24.8
_PRECISION_ = (3/4)
_PRECISION1_ = 24
_PRECISION2_= 0.00803982472

_TRAINING_SET_PATH_ = '/home/amirh/Documents/AIHOOSH/train.csv'
_TEST_SET_PATH_ = '/home/amirh/Documents/AIHOOSH/test.csv'
_STOP_WORDS_PATH_ = '/home/amirh/Documents/AIHOOSH/stopwords-fa.txt'
_RESULT_PATH_ = '/home/amirh/Documents/AIHOOSH/ans.csv'


# In[50]:


def make_oneColumn_set(data_frame : pd.DataFrame):
    data_frame[_TITLE_AND_COMMENT_] = data_frame[_TITLE_] + ' ' + data_frame[_COMMENT_]
    data_frame = data_frame.drop(columns = [_COMMENT_, _TITLE_] ,axis = 1)
    return data_frame


# In[51]:


def all_words_use(training_setX : pd.DataFrame, training_setY : pd.DataFrame):
    
    # a function to calculate all words used in data_frame
    using_words = dict() #first arg value , sec arg tuple
    
    number_count_valid = 0
    number_count_notValid = 0
    
    for index, row in training_setX.iterrows():
        words = str(row['title_comment']).split()
        
        if training_setY[index] == 0:#valid
            number_count_valid += len(words)
        else:
            number_count_notValid += len(words)
        
        for word in words:
            if word not in using_words.keys(): #new word stat if not in dict
                if training_setY[index] == 0:    #if valid
                    using_words.update({word : (1, 0)}) #update dict of valids to 1
                elif training_setY[index] == 1:
                    using_words.update({word : (0, 1)}) #
            else:
                old_tuple = using_words[word]
                if training_setY[index] == 0:
                    new_tuple = (old_tuple[0] + 1, old_tuple[1])
                    using_words.update({word : new_tuple})
                elif training_setY[index] == 1:
                    new_tuple = (old_tuple[0], old_tuple[1] + 1)
                    using_words.update({word : new_tuple})
        
    return using_words, number_count_valid, number_count_notValid


# In[52]:


def delete_stop_words(data_frame : pd.DataFrame):
    stop_words = pd.read_csv(_STOP_WORDS_PATH_)
    data_frame[_TITLE_AND_COMMENT_].apply(lambda x: [item for item in data_frame if item not in stop_words])
    return data_frame


# In[53]:


def prediction_valid_words(words : dict, number_count_valid, number_count_notValid):
    predictions = dict()
    
    number_uniuqe_words = len(words.keys())
    
    for key, value in words.items():
        p_word_valid = (value[0] + _CONST_) / (number_count_valid + number_uniuqe_words)
        p_word_notValid = (value[1] + _CONST_) / (number_count_notValid + number_uniuqe_words)
        predictions.update({key : (p_word_valid, p_word_notValid)})
    
    return predictions


# In[54]:


def predict_value(title_comment : str, predictions : dict, p_valid):
    words = title_comment.split()
    p_notValid = 1 - p_valid + _PRECISION_ 
    for word in words:
        if word in predictions.keys():
            p_valid *= _PRECISION1_*predictions[word][0]*_PRECISION1_
            p_notValid *= _PRECISION1_*predictions[word][1]*_PRECISION1_
        else:
            p_valid *= _PRECISION2_
            p_notValid *= _PRECISION2_
    return (0 if p_valid >= p_notValid else 1)


# In[57]:


#Main

# clear training set and test set
training_set = pd.read_csv(_TRAINING_SET_PATH_)[[_TITLE_ , _COMMENT_ , _VERIF_STAT_]]
test_set     = pd.read_csv(_TEST_SET_PATH_)[[_ID_ , _TITLE_ , _COMMENT_]]

# # delete rows which contains nan values
# training_set = training_set.dropna(axis = 'rows')
# test_set = test_set.dropna(axis = 'rows')

# split to training_setX and training_setY
training_setX = training_set[[_TITLE_ , _COMMENT_]]
training_setY = training_set[_VERIF_STAT_]

test_setX = test_set[[_TITLE_, _COMMENT_]]

# append column title to column comment
training_setX = make_oneColumn_set(training_setX)
test_setX    = make_oneColumn_set(test_setX)

# delete punctuation from our training set
training_setx = training_setX[_TITLE_AND_COMMENT_].str.replace('[^\w\s]','')
test_setx = test_setX[_TITLE_AND_COMMENT_].str.replace('[^\w\s]','')

training_setX[_TITLE_AND_COMMENT_] = delete_stop_words(training_setX)
test_setX[_TITLE_AND_COMMENT_] = delete_stop_words(test_setX)

p_valid = len(training_setY[training_setY == 0]) / len(training_setY)


# In[58]:


using_words, number_cValid, number_cNotValid = all_words_use(training_setX, training_setY)


# In[59]:


predictions = prediction_valid_words(using_words, number_cValid, number_cNotValid )


# In[60]:


results = pd.DataFrame(columns = [_ID_ , _TEST_SET_PATH_])
for index, row in test_setX.iterrows():
    pred = predict_value(str(row[ _TITLE_AND_COMMENT_ ]) , predictions , p_valid)
    results.loc[index] = [test_set[_ID_][index], pred]


# In[61]:


results.to_csv(_RESULT_PATH_)


# In[ ]:





# In[ ]:




