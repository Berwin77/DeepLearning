
# coding: utf-8

# In[1]:


import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')
X_train = np.array(X_train) 
y_train = np.array(y_train) 
X_test = np.array(X_test) 
y_test = np.array(y_test) 





def normalize(x):
    y = (x-np.min(x))/(np.max(x)-np.min(x))
    return y

X_train = normalize(X_train)
X_test = normalize(X_test)
print(X_train[:5])

from sklearn import preprocessing
# from keras import backend as K
# y_train = K.one_hot(y_train, 10)
# from keras.utils import np_utils   
# y_train_ohe = np_utils.to_categorical(y_train)  
def one_hot_encode(x):
    lb = preprocessing.LabelBinarizer()
    lb.fit(range(10))
    return lb.transform(x)
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)
#print(y_train.shape)
#print(y_train[:5])

X_train = X_train.reshape(-1, 28, 28,1).astype('float32')  
X_test = X_test.reshape(-1,28, 28,1).astype('float32')  


# In[4]:



from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import get_file
## decode_predictions 输出5个最高概率：(类名, 语义概念, 预测概率) decode_predictions(y_pred)
from keras.applications.imagenet_utils import decode_predictions

#  预处理 图像编码服从规定，譬如,RGB，GBR这一类的，preprocess_input(x)  
from keras.applications.imagenet_utils import _obtain_input_shape

#确定适当的输入形状，相当于opencv中的read.img，将图像变为数组
from keras.engine.topology import get_source_inputs




def ConvNetwork(classes=10):
    
    img_input = Input(shape=(28, 28, 1))
    #Block1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    #block2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)

    
    x = Flatten(name='flatten')(x)
    #FC1
    x = Dense(1000, name='fc1_d')(x)
    x = Activation('relu', name='fc1_a')(x)
    #FC2
    x = Dense(500, name='fc2_d')(x)
    x = Activation('relu', name='fc2_a')(x)

    #softmax
    x = Dense(classes, name='soft_max1')(x)
    predictions = Activation('softmax', name='soft_max2')(x)

    inputs = img_input
    model = Model(inputs, predictions, name='CNN')
    
    return model 




# In[9]:


if __name__ == '__main__':
    
    model = ConvNetwork(10)
    sgd = SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', 
    metrics=['accuracy'])  
    model.fit(X_train, y_train, epochs=10, batch_size=32,shuffle=True)
    model.evaluate(X_test, y_test, verbose=0)


#  
