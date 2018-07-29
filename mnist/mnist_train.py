import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = 'path/to/model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    ### 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ###在训练神经网络模型时，每过一遍数据需要反向传播来更新神经网络参数，又要更新每一个参数的滑动平均值。为了完成多个操作
    ###tensorflow提供了tf.control_dependencies和tf.group两种机制。下面两行程序是等价的：
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    ###等价于 train_op = tf.group(train_step, variables_avrages_op)
    saver = tf.train.Saver()
    # 计算在当前 batch 中所有样例的交叉熵平均值
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ###训练和测试 验证 分离
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print('After %d training steps, loss on training batch is %g ' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('path/to/mnist_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()