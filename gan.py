import glob
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt
from mobilenet_v1 import mobilenet_v1
from colorutil import *
from model import generator, discriminator
from ops import resize, conv2d, fc

tf.app.flags.DEFINE_boolean('train', True, 'is train mode?')
FLAGS = tf.app.flags.FLAGS

# GAN Hyperparameter
_lambda = 0.001
gamma = 0.95

if FLAGS.train == True:
  filenames = sorted(glob.glob('./images/img/*.jpg'))
  batch_size = 5
  epoch = 2
else:
  filenames = sorted(glob.glob('./test_images/*.jpg'))
  batch_size = 1
  epoch = 1

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
  global FLAGS
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=False)
  example = read_my_file_format(filename_queue, randomize=False)
  if FLAGS.train == True:
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
  else:
    min_after_dequeue = 0
    capacity = 1
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

#initializer = tf.random_normal_initializer(stddev=10.0)
#initializer = tf.truncated_normal_initializer(stddev=0.1)
initializer = tf.contrib.layers.xavier_initializer()
#inputs = tf.placeholder(tf.float32, shape=[32, 224, 224, 3])
inputs = input_pipeline(filenames, batch_size, num_epochs=epoch)
#grayscale_one, _ = tf.split(rgb_to_lab(grayscale), [1, 2], axis=3)
#
#yuv_images = rgb_to_lab(inputs)
#yuv_y, u, v = tf.split(yuv_images, [1, 1, 1], axis=3)

r2l_images = rgb_to_lab(inputs)

grayscale = tf.image.rgb_to_grayscale(inputs)
grayscale = tf.concat([grayscale, grayscale, grayscale], axis=3)
grayscale_lab = rgb_to_lab(grayscale)
global_step = tf.Variable(0, trainable=False, name='global_step')

#_, w = mobilenet_v1(grayscale, is_training=False)

is_training = True

learn_rate = tf.maximum(tf.train.exponential_decay(1e-2, global_step, 300, 0.95, staircase=True), 1e-5)

comp_out = generator(None, grayscale_lab)
net_rgb = lab_to_rgb(comp_out)

dis_image, dis_error = discriminator(r2l_images, reuse=False)
gen_image, gen_error = discriminator(comp_out, reuse=True)

dis_image_rgb = lab_to_rgb(dis_image)
gen_image_rgb = lab_to_rgb(gen_image)

k = tf.Variable(0., trainable=False)
update_k = k.assign(tf.clip_by_value(k + _lambda*(gamma*dis_error - gen_error), 0, 1))

#gen_error += k*disc(grayscale_lab, reuse=True)[1]
loss_1 = gen_error
loss_2 = dis_error - k*gen_error

M = dis_error + tf.abs(gamma*dis_error - gen_error)

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'generator' in var.name]
d_vars = [var for var in t_vars if 'discriminator' in var.name]

optimizer_g = tf.train.AdamOptimizer(learn_rate*5, beta1=0.5)
optimizer_d = tf.train.AdamOptimizer(learn_rate, beta1=0.5)

op1 = optimizer_g.minimize(loss_1, var_list=g_vars)
op2 = optimizer_d.minimize(loss_2, var_list=d_vars)

# incr
incr = tf.assign(global_step, global_step+1)

r2l_mean = tf.reduce_mean(r2l_images)
net_mean = tf.reduce_mean(comp_out)
#loss = tf.reduce_mean(loss)

with tf.Session() as sess:
  if FLAGS.train == True:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    var_list = []
    for x in slim.get_model_variables():
      if not ("MobilenetV1/AuxLogits" in x.op.name or "MobilenetV1/Logits" in x.op.name or "MobilenetV1" not in x.op.name):
        var_list.append(x)
    mobilenet_restore = slim.assign_from_checkpoint_fn('checkpoints/mobilenet_v1_1.0_224.ckpt', var_list, ignore_missing_vars=True)
    mobilenet_restore(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver = tf.train.Saver()
    #saver.restore(sess, './checkpoints/model.ckpt')
    #images = input_pipeline(filenames, 32, num_epochs=10000)

    try:
      step = 0
      while not coord.should_stop():
        result = sess.run([op1, op2, update_k])
        result = sess.run([incr, global_step])
        step = result[-1]

        if step % 333 == 1:
          result_image = sess.run([inputs, grayscale, net_rgb, dis_image_rgb, gen_image_rgb])
          step_view = concat_images(result_image[1][0], result_image[0][0])
          step_view = concat_images(step_view, result_image[2][0])
          step_view = concat_images(step_view, result_image[3][0])
          step_view = concat_images(step_view, result_image[4][0])
          plt.imsave("summary_gan_1/" + str(step) + ".jpg", step_view)

          result_summary = sess.run([M, learn_rate, gen_error, dis_error, k])
          print(f'step: {step}, M: {result_summary[0]}, learn_rate: {result_summary[1]}, l_gen: {result_summary[2]}, l_disc: {result_summary[3]}, k: {result_summary[4]}')
          if result_summary[3] < result_summary[2]:
            print('equilibrium broken')
          #print(f'step: {step}, M: {result_image[4]}, learn_rate: {result_image[5]}, l1: {result_image[6]}, l2: {result_image[7]}, k: {result_image[8]}')

        if step % 10000 == 1:
          saver = tf.train.Saver()
          saver.save(sess, "./checkpoints/model_gan1.ckpt")
    except tf.errors.OutOfRangeError:
      saver = tf.train.Saver()
      saver.save(sess, "./checkpoints/model_gan1.ckpt")
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()

    coord.join()
  else:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #sess.run([tf.local_variables_initializer()])

    #var_list = []
    #for x in slim.get_model_variables():
    #  if not ("MobilenetV1/AuxLogits" in x.op.name or "MobilenetV1/Logits" in x.op.name or "MobilenetV1" not in x.op.name):
    #    var_list.append(x)
    #mobilenet_restore = slim.assign_from_checkpoint_fn('checkpoints/mobilenet_v1_1.0_224.ckpt', var_list, ignore_missing_vars=True)
    #mobilenet_restore(sess)

    saver = tf.train.Saver()
    saver.restore(sess, './checkpoints/model_gan1.ckpt')

    try:
      step = 0
      while not coord.should_stop():
        step += 1
        result_image = sess.run([inputs, grayscale, net_rgb])
        step_view = result_image[1][0]
        step_view = concat_images(step_view, result_image[0][0])
        step_view = concat_images(step_view, result_image[2][0])
        plt.imsave("outputs/" + str(step) + ".jpg", step_view)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      coord.request_stop()

    coord.join()
