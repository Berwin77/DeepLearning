
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):###regularizer是正则化函数
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
    if regularizer!=None:
        ###自定义集合losses 正则化的weights存入该集合
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE],
                                      regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE],
                                 regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2

