

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

def build_inverse_mapping(measurement):
    patch_est = tf.layers.dense(measurement, blockSize ** 2, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01, dtype=tf.float64), name='inverse')
    return patch_est

def build_forward_mapping(patch):
    measurement_est = tf.layers.dense(patch, int(np.ceil(measurement_rate * (blockSize ** 2))), use_bias=False, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01, dtype=tf.float64), name='forward')
    return measurement_est

# define loss function for model
def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)

    return loss

# create tf dataset from input data
dataset = tf.contrib.data.Dataset.from_tensor_slices((measurements, patches))

dataset = dataset.repeat(100)
dataset = dataset.batch(100) 

# initialize iterator
iterator = dataset.make_initializable_iterator()

next_measurement, next_patch = iterator.get_next()

#patch_est, A_t = build_linear_mapping(next_measurement)
measurement_est = build_forward_mapping(next_patch)
patch_est = build_inverse_mapping(measurement_est)

loss = build_loss(patch_est, next_patch)

# define optimization procedure
training_op = tf.train.AdagradOptimizer(5e-3).minimize(loss)

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

            phi_inv = tf.get_default_graph().get_tensor_by_name('inverse' + '/kernel:0')
            phi = tf.get_default_graph().get_tensor_by_name('forward' + '/kernel:0')
            
            sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv.mat', {'phi_inv':sess.run(phi_inv)})
            sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_est.mat', {'phi':sess.run(phi)})

            #print(sess.run(weights))


            break