
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import scipy.io as sio
import os

f = sio.loadmat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/train_dataset')

# set block_size and measurement_rate
block_size = 16
measurement_rate = 0.25

# load measurements and patches dataset
measurements = f['measurements']
measurements = np.squeeze(measurements).transpose([1,0])
patches = f['patches_vec']
patches = np.squeeze(patches).transpose([1,0])

# define linear model to train
def build_linear_mapping(measurement):
    phi_inv = tf.Variable(tf.random_normal([int(np.ceil(measurement_rate * (block_size ** 2))), (block_size ** 2)], mean = 0.0, stddev=0.01, dtype=tf.float64))

    patch_est = tf.matmul(measurement, phi_inv) 

    return patch_est, phi_inv

# define loss function for model
def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)

    return loss

# create tf dataset from input data
dataset = tf.contrib.data.Dataset.from_tensor_slices((measurements, patches))

dataset = dataset.repeat(1)
dataset = dataset.batch(100) 

# initialize iterator
iterator = dataset.make_initializable_iterator()

next_measurement, next_patch = iterator.get_next()

patch_est, phi_inv = build_linear_mapping(next_measurement)
loss = build_loss(patch_est, next_patch)

# define optimization procedure
training_op = tf.train.GradientDescentOptimizer(2e-2).minimize(loss)

with tf.Session() as sess:
    # initialize iterator and global variables
    sess.run(iterator.initializer)
    tf.global_variables_initializer().run(session=sess)

    while True:
        try:
            sess.run(training_op)
            print(sess.run(loss))

        except tf.errors.OutOfRangeError:
            print(sess.run(phi_inv))
            sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv.mat', {'phi_inv':sess.run(phi_inv)})

            break