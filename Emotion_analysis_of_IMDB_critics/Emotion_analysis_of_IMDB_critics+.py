
# coding: utf-8

# In[1]:



import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
#从imdb数据库下载数据，选择路径path，n_word代表多少个单词作为词袋，验证集占比为valid_portion
train, test,_ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train 
testX, testY = test

#转换成向量
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#将label进行one-hot
trainY = to_categorical(trainY, nb_classes=2)
testY= to_categorical(testY, nb_classes=2)

#创建网络

net = tflearn.input_data([None, 100])

#如果有些离散对象自然被编码为离散的原子，例如独特的ID，它们不利于机器学习的使用和泛化。
#可以理解embedding是将非矢量对象转换为机器学习处理的输入。
#将100个输入转换为10000个id，再转化为128维的连续向量
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
#lstm层  drop防止过拟合
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net , 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
#构建模型
# tensorboard需要的日志文件存储在/tmp/tflearn_logs中
model = tflearn.DNN(net)
#训练模型
model.fit(trainX,trainY, show_metric=True,batch_size=32)

