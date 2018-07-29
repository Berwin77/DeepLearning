import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train
###每秒加载一次最新模型
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ### 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数求平均值了。
        ### 可以共用mnist_inference.py中定义的来进行前向传播过程。
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        ## 每隔ECAL_INTERVAL_SECS 秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps , validation accuracy = %g" %(global_step, accuracy_score))
                else:
                    print('NO checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets('path/to/mnist_data', one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()