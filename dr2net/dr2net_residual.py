
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import scipy.io as sio
import os

f = sio.loadmat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/train_dataset')

# set blockSize and measurement_rate
blockSize = 16
measurement_rate = 0.25

# load measurements and patches dataset
measurements = f['measurements']
measurements = np.squeeze(measurements).transpose([1,0])
patches = f['patches_vec']
patches = np.squeeze(patches).transpose([1,0])

def build_linear_mapping(measurement):
    patch_est_linear = tf.layers.dense(measurement, blockSize ** 2, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01, dtype=tf.float64), name='fc1')

    return patch_est_linear

def build_resnet(patch_est):
    patch_est = tf.reshape(patch_est, shape=[-1, blockSize, blockSize, 1])

    patch_est = tf.cast(patch_est, tf.float32)

    conv1 = tf.layers.conv2d(patch_est, filters=64, kernel_size=11, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=1, strides=(1,1), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=1, kernel_size=7, strides=(1,1), padding='same')

    patch_est_residual = tf.reshape(conv3, [-1, 256])
    patch_est_residual = tf.layers.dense(patch_est_residual, 256, use_bias=False, name = 'fc2')

    return patch_est_residual

# define loss function for model
def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)

    return loss

# create tf dataset from input data
dataset = tf.contrib.data.Dataset.from_tensor_slices((measurements, patches))

dataset = dataset.repeat(10)
dataset = dataset.batch(1000) 

# initialize iterator
iterator = dataset.make_initializable_iterator()

next_measurement, next_patch = iterator.get_next()

patch_est_linear = build_linear_mapping(next_measurement)
patch_est_residual = build_resnet(patch_est_linear)

loss = build_loss(next_patch, patch_est_residual)

learning_rate = 0.01

# define optimization procedure
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # initialize iterator and global variables
    sess.run(iterator.initializer)
    tf.global_variables_initializer().run(session=sess)

    while True:
        try:
            sess.run(training_op)
            print(sess.run(loss))


        except tf.errors.OutOfRangeError:
            #print(sess.run(A_t))
            #sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/A_t.mat', {'A_t':sess.run(A_t)})

            phi_inv = tf.get_default_graph().get_tensor_by_name('fc1' + '/kernel:0')
            sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv1.mat', {'phi_inv1':sess.run(phi_inv)})

            phi_inv = tf.get_default_graph().get_tensor_by_name('fc2' + '/kernel:0')
            sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv2.mat', {'phi_inv2':sess.run(phi_inv)})

            print(sess.run(phi_inv))

            break