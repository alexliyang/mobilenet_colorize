import tensorflow as tf
from ops import conv2d, deconv2d, fc, resize

def generator(x, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    l, _ = tf.split(x, [1, 2], axis=3)

    net = conv2d(x, 3, 32, scope='conv_1')
    net = conv2d(net, 32, 64, scope='conv_2')
    net = conv2d(net, 64, 128, scope='conv_3')
    net = conv2d(net, 128, 2, scope='deconv_4', bn=False)
    net = tf.reshape(net, [-1, 224, 224, 2])

    net = tf.concat([l, net], axis=3)

    return net
