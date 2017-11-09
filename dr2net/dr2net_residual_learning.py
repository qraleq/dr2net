
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import scipy.io as sio
import os

import matplotlib.pyplot as plt
from skimage import color, io, filters

# set blockSize and measurement_rate
blockSize = 16
measurement_rate = 0.25

# load measurements and patches from training dataset
f = sio.loadmat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/training_dataset')

training_measurements = f['measurements']
training_measurements = np.squeeze(training_measurements).transpose([1,0])
training_patches = f['patches_vec']
training_patches = np.squeeze(training_patches).transpose([1,0])

# load measurements and patches from validation dataset
f = sio.loadmat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/training_dataset')

validation_measurements = f['measurements']
validation_measurements = np.squeeze(validation_measurements).transpose([1,0])
validation_patches = f['patches_vec']
validation_patches = np.squeeze(validation_patches).transpose([1,0])


# define linear mapping operation using one fully connected layer
def build_linear_mapping(measurement):
    patch_est = tf.layers.dense(measurement, blockSize ** 2, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.0005, dtype=tf.float64), name='fc')

    return patch_est

def build_resnet(patch_est):
    patch_est = tf.reshape(patch_est, shape=[-1, blockSize, blockSize, 1])

    patch_est = tf.cast(patch_est, tf.float32)

    conv1 = tf.layers.conv2d(patch_est, filters=64, kernel_size=11, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=1, strides=(1,1), padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=1, kernel_size=7, strides=(1,1), padding='same')

    patch_est_residual = tf.reshape(conv3, [-1, 256])
    patch_est_residual = tf.cast(patch_est_residual, tf.float64)

    return patch_est_residual

# define loss function for model
def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)

    return loss


# create tf dataset from input data
training_dataset = tf.contrib.data.Dataset.from_tensor_slices((training_measurements, training_patches))
validation_dataset = tf.contrib.data.Dataset.from_tensor_slices((validation_measurements, validation_patches))

nEpochs = 50
training_dataset = training_dataset.batch(1000)
validation_dataset = validation_dataset.batch(3000)

# initialize iterator
iterator = Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

next_measurement, next_patch = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

patch_est_linear = build_linear_mapping(next_measurement)
loss_linear = build_loss(patch_est_linear, next_patch)

patch_est_resnet = build_resnet(patch_est_linear)
loss_resnet = build_loss(patch_est_resnet + patch_est_linear, next_patch)

joint_loss = loss_linear + loss_resnet

# define optimization procedure
learning_rate = 0.01
joint_op = tf.train.AdamOptimizer(learning_rate).minimize(joint_loss)
linear_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_linear)
resnet_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_resnet)


with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)

    for i in range(nEpochs):
        print('EPOCH NO. %d' % i)

        # initialize iterator
        sess.run(training_init_op)

        while True:
            try:
                sess.run(joint_op)

            except tf.errors.OutOfRangeError:
                break

        sess.run(validation_init_op)

        while True:
            try:
                print('VALIDATION LOSS: %f' % sess.run(joint_loss))

            except tf.errors.OutOfRangeError:
                break
    
    phi_inv = tf.get_default_graph().get_tensor_by_name('fc' + '/kernel:0')
    phi_inv = sess.run(phi_inv)

    sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv.mat', {'phi_inv':phi_inv})
    
    tf.train.Saver().save(sess, 'Y:/Projects/Python Projects/dr2net/dr2net/dataset/tmp/model.ckpt')

