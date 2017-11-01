
# coding: utf-8

# # 用TFLearn进行电影评价的情感分析
# 
# 

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


# 

# 

# In[6]:

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
reviews.head(10)


# 

# In[8]:

from collections import Counter

total_counts = Counter()
for i, row in (reviews.iterrows()):
    total_counts.update(row[0].split(" "))

print("Total words in data set: ", len(total_counts))


# ##  取total_counts的前10000个

# In[9]:

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])


# What's the last word in our vocabulary? We can use this to judge if 10000 is too few. If the last word is pretty common, we probably need to keep more words.

# In[10]:

print(vocab[-1], ': ', total_counts[vocab[-1]])


# ### 将词袋转换为向量

# In[75]:

word2idx = {word : i for i, word in enumerate(vocab)}


# ### 文本转换为向量
# 

# In[76]:

def text_to_vector(text):
    te2ve = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(" "):
        id = word2idx.get(word, None)
        if id == None:
            continue
        else:
            te2ve[id] += 1
                     
    return te2ve
                


# If you do this right, the following code should return
# 
# ```
# text_to_vector('The tea is for a party to celebrate '
#                'the movie so she has no time for a cake')[:65]
#                    
# array([0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
# ```       

# In[77]:

text_to_vector('The tea is for a party to celebrate '
               'the movie so she has no time for a cake')[:65]


# ### 将所有reviews转化为词袋向量

# In[80]:

word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])


# In[82]:

# Printing out the first 5 word vectors
word_vectors[:5, :23]


# ### 得到训练集和测试集

# In[83]:

Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)


# In[84]:

trainY


# # 搭建神经网络

# In[88]:

# Network building
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    #### Your code ####
    net = tflearn.input_data([None, 10000])
    
    net = tflearn.fully_connected(net, 200, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model


# ## 初始化模型

# In[89]:

model = build_model()


# ## 训练网络

# In[90]:

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=10)


# ## 进行测试

# In[91]:

predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)


# 

# In[ ]:




# In[ ]:



