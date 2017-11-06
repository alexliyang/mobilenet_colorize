import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt
from mobilenet_v1 import mobilenet_v1

batch_size = 10
filenames = sorted(glob.glob('./images/img/*.jpg'))

def read_my_file_format(filename_queue, randomize=False):
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=3)
    uint8image = tf.image.resize_images(uint8image, [224, 224])
    #uint8image = tf.random_crop(uint8image, (224, 224, 3))
    if randomize:
        uint8image = tf.image.random_flip_left_right(uint8image)
        uint8image = tf.image.random_flip_up_down(uint8image, seed=None)
    float_image = tf.div(tf.cast(uint8image, tf.float32), 255)
    return float_image

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=False)
    example = read_my_file_format(filename_queue, randomize=False)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp

def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.multiply(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp

initializer = tf.random_normal_initializer(stddev=0.01)
#inputs = tf.placeholder(tf.float32, shape=[32, 224, 224, 3])
inputs = input_pipeline(filenames, batch_size, num_epochs=10)
grayscale = tf.image.rgb_to_grayscale(inputs)
grayscale = tf.concat([grayscale, grayscale, grayscale], axis=3)
#yuv_images = rgb2yuv(inputs)
#y, uv = tf.split(yuv_images, [1, 2], axis=3)
_, w = mobilenet_v1(grayscale, is_training=True)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def resize(x, shape):
  return tf.image.resize_images(x, shape)

# 7 x 7 x 1024 -> 14 x 14 x 512
global_step = tf.Variable(0, trainable=False, name='global_step')

W1 = tf.get_variable('weight1', [3, 3, 1024, 512], initializer=initializer)
W2 = tf.get_variable('weight2', [3, 3, 512, 256], initializer=initializer)
W3 = tf.get_variable('weight3', [3, 3, 256, 128], initializer=initializer)
W4 = tf.get_variable('weight4', [3, 3, 128, 64], initializer=initializer)
W5 = tf.get_variable('weight5', [3, 3, 64, 32], initializer=initializer)
W6 = tf.get_variable('weight6', [1, 1, 32, 3], initializer=initializer)

x = conv2d(w['Conv2d_13_pointwise'], W1)
x = conv2d(resize(x, (14, 14)) + w['Conv2d_11_pointwise'], W2)
x = conv2d(resize(x, (28, 28)) + w['Conv2d_5_pointwise'], W3)
x = conv2d(resize(x, (56, 56)) + w['Conv2d_3_pointwise'], W4)
x = conv2d(resize(x, (112, 112)) + w['Conv2d_1_pointwise'], W5)
x = conv2d(resize(x, (224, 224)), W6) + grayscale

out = x

#W2 = tf.get_variable('weight2', [3, 3, 64, 3])
#x2 = tf.image.resize_images(w['Conv2d_1_pointwise'], (224, 224))
#out2 = conv2d(x2, W2)
#
#out3 = out + out2
#W3 = tf.get_variable('weight3', [1, 1, 3, 3])
#out4 = conv2d(out3 + grayscale, W3)

net = tf.maximum(out, tf.zeros_like(out))
net = tf.minimum(net, tf.ones_like(out))

#uv_pred_to_image = yuv2rgb(tf.concat([y, net], axis=3))
#y_to_image = tf.image.rgb_to_grayscale(inputs)
#grayscale = tf.concat([grayscale, grayscale, grayscale], axis=3)
#loss = tf.losses.mean_pairwise_squared_error(inputs, out) + tf.nn.moments(out, axes=[0,1,2])[1]
loss = tf.losses.mean_pairwise_squared_error(rgb2yuv(inputs), rgb2yuv(out))
learn_rate = tf.train.exponential_decay(1e-3, global_step, 500, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learn_rate)
params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

  #W = tf.reshape(W, [1024, 1024])
#for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#  print(i)
with tf.Session() as sess:
  sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
  saver = tf.train.Saver()
  #saver.restore(sess, 'checkpoints/mobilenet_v1_1.0_224.ckpt')
  var_list = []
  for x in slim.get_model_variables():
    if not ("MobilenetV1/AuxLogits" in x.op.name or "MobilenetV1/Logits" in x.op.name or "MobilenetV1" not in x.op.name):
      var_list.append(x)
  sexy = slim.assign_from_checkpoint_fn('checkpoints/mobilenet_v1_1.0_224.ckpt', var_list, ignore_missing_vars=True)
  sexy(sess)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  #images = input_pipeline(filenames, 32, num_epochs=10000)

  try:
    step = 0
    while not coord.should_stop():
      result = sess.run([op, loss, global_step])
      step = result[2]
      if step % 317 == 1:
        result_image = sess.run([inputs, grayscale, net])
        step_view = concat_images(result_image[1][0], result_image[0][0])
        step_view = concat_images(step_view, result_image[2][0])
        print(f'step: {step}, loss: {result[1]}')
        plt.imsave("summary/" + str(step) + "_0.jpg", step_view)
      if step % 10000 == 0:
        saver.save(sess, "./checkpoints/model.ckpt")
  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    coord.request_stop()

  coord.join()
  sess.close()
