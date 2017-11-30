import tensorflow as tf

def resize(x, shape):
  return tf.image.resize_images(x, shape)

def lrelu(x, a=0.2):
  with tf.name_scope("lrelu"):
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def conv2d(x, in_size, out_size, k=3, d=1, bn=True, scope='conv2d'):
  with tf.variable_scope(scope):
    W = tf.get_variable('Weight', [k, k, in_size, out_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Bias", shape=out_size,
        initializer=tf.constant_initializer(0.))

    net = tf.nn.conv2d(x, W, strides=[1, d, d, 1], padding="SAME")
    net = tf.nn.bias_add(net, b)

    if bn == True:
      net = tf.contrib.layers.batch_norm(net)
      net = lrelu(net)

    return net

def deconv2d(x, out_size, k=3, d=1, bn=True, scope='deconv2d'):
  with tf.variable_scope(scope):
    batch_size = tf.to_int32(tf.shape(x)[0])
    h = tf.to_int32(tf.shape(x)[1])
    w = tf.to_int32(tf.shape(x)[2])
    in_size = tf.to_int32(tf.shape(x)[3])

    output_shape = tf.stack([batch_size, h, w, tf.Dimension(out_size)])
    print(output_shape)

    W = tf.get_variable('Weight', [k, k, out_size, in_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("Bias", shape=out_size,
        initializer=tf.constant_initializer(0.))

    net = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, d, d, 1])
    net = tf.nn.bias_add(net, b)

    if bn == True:
      net = tf.contrib.layers.batch_norm(net)
      net = lrelu(net)

    return net

def fc(x, out_size, scope='fc'):
  with tf.variable_scope(scope):
    b = tf.get_variable("Bias", shape=out_size,
        initializer=tf.constant_initializer(0.))

    net = tf.contrib.layers.fully_connected(x, out_size, activation_fn=None)
    net = tf.nn.bias_add(net, b)

    return net
