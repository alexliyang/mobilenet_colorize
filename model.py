import tensorflow as tf
from ops import conv2d, deconv2d, fc, resize

def discriminator(x, reuse=False):
  with tf.variable_scope('discriminator', reuse=reuse):
    net = resize(x, (56, 56))
    net = conv2d(net, 3, 32, scope='conv_1')
    net = conv2d(net, 32, 64, scope='conv_2')
    net = tf.reshape(net, [-1, 56 * 56 * 64])
    code = fc(net, 16, scope='fc_3')
    net = fc(code, 56 * 56 * 64, scope='fc_4')
    net = tf.reshape(net, [-1, 56, 56, 64])
    net = conv2d(net, 64, 32, scope='conv_5')
    net = conv2d(net, 32, 3, scope='conv_6', bn=False)
    net = resize(x, (224, 224))

    loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)

    return net, loss

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
