import tensorflow as tf
from colorutil import lab_to_rgb
from ops import conv2d, deconv2d, fc, resize
from mobilenet_v1 import mobilenet_v1

def mobilenet(x, reuse=False):
  _, w = mobilenet_v1(x, is_training=False, reuse=reuse)
  return w

def discriminator_ae(x, reuse=False):
  with tf.variable_scope('discriminator', reuse=reuse):
    net = conv2d(x, 3, 32, d=2, scope='conv_1')
    net = conv2d(net, 32, 64, d=2, scope='conv_2')
    net = tf.reshape(net, [-1, 56 * 56 * 64])
    code = fc(net, 16, scope='fc_3')
    net = fc(code, 56 * 56 * 64, scope='fc_4')
    net = tf.reshape(net, [-1, 56, 56, 64])
    net = resize(net, (112, 112))
    net = conv2d(net, 64, 32, scope='conv_5')
    net = resize(net, (224, 224))
    net = deconv2d(net, 32, 16, 224, scope='conv_6')

    dimension = 2

    if dimension == 2:
      sc_1, x = tf.split(x, [1, 2], axis=3)
      net = deconv2d(net, 16, 2, 224, scope='deconv_7', bn=False)
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)
      net = tf.concat([sc_1, net], axis=3)
    else:
      net = deconv2d(net, 16, 3, 224, scope='deconv_10', bn=False)
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)

    return net, loss

def generator_with_infer(infer, x, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    l, _ = tf.split(x, [1, 2], axis=3)

    net = conv2d(infer, 3, 32, scope='conv_1')
    net = conv2d(net, 32, 64, scope='conv_2')
    net = conv2d(net, 64, 128, scope='conv_3')
    net = conv2d(net, 128, 2, scope='deconv_4', bn=False)
    net = tf.reshape(net, [-1, 224, 224, 2])

    net = tf.concat([l, net], axis=3)

    return net

def generator_without_infer(x, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    l, _ = tf.split(x, [1, 2], axis=3)

    net = conv2d(x, 3, 32, scope='conv_1')
    net = conv2d(net, 32, 64, scope='conv_2')
    net = conv2d(net, 64, 128, scope='conv_3')
    net = conv2d(net, 128, 2, scope='deconv_4', bn=False)
    net = tf.reshape(net, [-1, 224, 224, 2])

    net = tf.concat([l, net], axis=3)

    return net

def deep_generator_without_infer(x, reuse=False):
  with tf.variable_scope('generator', reuse=reuse):
    l, x = tf.split(x, [1, 2], axis=3)

    net = conv2d(x, 2, 32, d=2, scope='conv_1')
    net = conv2d(net, 32, 64, d=2, scope='conv_2')
    net = tf.reshape(net, [-1, 56 * 56 * 64])
    code = fc(net, 16, scope='fc_3')
    net = fc(code, 56 * 56 * 64, scope='fc_4')
    net = tf.reshape(net, [-1, 56, 56, 64])
    net = resize(net, (112, 112))
    net = conv2d(net, 64, 32, scope='conv_5')
    net = resize(net, (224, 224))
    net = deconv2d(net, 32, 2, 224, scope='conv_6', bn=False)
    net = tf.concat([l, net], axis=3)

    return net

def generator_mobilenet(x, reuse=False):
  rgb_image = lab_to_rgb(x)
  #infer = lab_to_rgb(infer)
  w = mobilenet(rgb_image, reuse=False)

  with tf.variable_scope('generator', reuse=reuse):
    net = deconv2d(w['Conv2d_13_pointwise'], 1024, 512, 7, scope='deconv_1')
    net = conv2d(resize(net, (14, 14)) + w['Conv2d_11_pointwise'], 512, 512, scope='deconv_2')
    net = deconv2d(net + w['Conv2d_9_pointwise'], 512, 512, 14, scope='deconv_3')
    net = conv2d(net + w['Conv2d_7_pointwise'], 512, 512, scope='deconv_4')
    net = deconv2d(net, 512, 256, 14, scope='deconv_5')
    net = conv2d(resize(net, (28, 28)) + w['Conv2d_5_pointwise'], 256, 128, scope='deconv_6')
    net = deconv2d(resize(net, (56, 56)) + w['Conv2d_3_pointwise'], 128, 64, 56, scope='deconv_7')
    net = conv2d(resize(net, (112, 112)) + w['Conv2d_1_pointwise'], 64, 32, scope='deconv_8')
    net = deconv2d(resize(net, (224, 224)), 32, 16, 224, scope='deconv_9', bn=False)

    dimension = 2

    if dimension == 2:
      sc_1, x = tf.split(x, [1, 2], axis=3)
      net = conv2d(net, 16, 2, scope='conv_10', bn=False)
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)
      net = tf.concat([sc_1, net], axis=3)
    else:
      net = deconv2d(net, 16, 3, 224, scope='deconv_10', bn=False)
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)

    return net

def generator(infer, x, reuse=False):
  return generator_mobilenet(x, reuse)

def discriminator_mobilenet(x, reuse=False):
  rgb_image = lab_to_rgb(x)

  grayscale = tf.image.rgb_to_grayscale(rgb_image)
  grayscale = tf.concat([grayscale, grayscale, grayscale], axis=3)

  w = mobilenet(grayscale, reuse=True)

  with tf.variable_scope('discriminator', reuse=reuse):
    net = deconv2d(w['Conv2d_13_pointwise'], 1024, 512, 7, scope='deconv_1')
    net = conv2d(resize(net, (14, 14)) + w['Conv2d_11_pointwise'], 512, 512, scope='deconv_2')
    net = deconv2d(net + w['Conv2d_9_pointwise'], 512, 512, 14, scope='deconv_3')
    net = conv2d(net + w['Conv2d_7_pointwise'], 512, 512, scope='deconv_4')
    net = deconv2d(net, 512, 256, 14, scope='deconv_5')
    net = conv2d(resize(net, (28, 28)) + w['Conv2d_5_pointwise'], 256, 128, scope='deconv_6')
    net = deconv2d(resize(net, (56, 56)) + w['Conv2d_3_pointwise'], 128, 64, 56, scope='deconv_7')
    net = conv2d(resize(net, (112, 112)) + w['Conv2d_1_pointwise'], 64, 32, scope='deconv_8')
    net = deconv2d(resize(net, (224, 224)), 32, 16, 224, scope='deconv_9')

    dimension = 2

    if dimension == 2:
      sc_1, x = tf.split(x, [1, 2], axis=3)
      net = conv2d(net, 16, 2, scope='conv_10', bn=False)
      #W = tf.get_variable('L2Weight', [224, 224, 2], initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
      #loss = tf.reduce_mean(
      #    tf.map_fn(lambda xn: tf.sqrt(2*tf.nn.l2_loss(xn * W))/(224*224), (x-net)))
      loss_net = tf.nn.dropout(net, 0.5) * 0.5
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - loss_net)) / (224*224)
      #loss = tf.sqrt(tf.reduce_sum(tf.image.total_variation(x - net))) / int(x.shape[0])
      net = tf.concat([sc_1, net], axis=3)
    else:
      net = deconv2d(net, 16, 3, 224, scope='deconv_10', bn=False)
      loss = tf.sqrt(2 * tf.nn.l2_loss(x - net)) / (224*224)

  color_encoder = False
  if color_encoder == True:
    _lambda = 1e-4
    loss += _lambda * tf.reduce_sum(tf.image.total_variation(net)) / int(x.shape[0])
    #w2 = mobilenet(rgb_image, reuse=True)
    #loss += tf.sqrt(2 * tf.nn.l2_loss(w['Conv2d_13_pointwise'] - w2['Conv2d_13_pointwise'])) / (7*7)

  return net, loss

def discriminator(x, reuse=False):
  logit, loss = discriminator_mobilenet(x, reuse)
  return logit, loss
