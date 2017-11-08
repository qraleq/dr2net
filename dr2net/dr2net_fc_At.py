
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
import scipy.io as sio
import os


# set block_size and measurement_rate
block_size = 16
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
    patch_est = tf.layers.dense(measurement, block_size ** 2, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.0005, dtype=tf.float64), name='fc')
    return patch_est

# define loss function for model
def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)

    return loss

# create tf dataset from input data
training_dataset = tf.contrib.data.Dataset.from_tensor_slices((training_measurements, training_patches))
validation_dataset = tf.contrib.data.Dataset.from_tensor_slices((validation_measurements, validation_patches))

nEpochs = 10
training_dataset = training_dataset.batch(1000)
validation_dataset = validation_dataset.batch(100)

# initialize iterator
iterator = Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

next_measurement, next_patch = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

patch_est = build_linear_mapping(next_measurement)
loss = build_loss(patch_est, next_patch)

# define optimization procedure
learning_rate = 0.01
learning_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)

    for i in range(nEpochs):
        print('EPOCH NO. %d'%i)

        # initialize iterator
        sess.run(training_init_op)

        while True:
            try:
                sess.run(learning_op)

            except tf.errors.OutOfRangeError:
                #print(sess.run(A_t))
                #sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/A_t.mat', {'A_t':sess.run(A_t)})



                #print(sess.run(phi_inv))


                break

        sess.run(validation_init_op)

        while True:
            try:
                print('VALIDATION LOSS: %f'%sess.run(loss))

            except tf.errors.OutOfRangeError:
                break
    
    phi_inv = tf.get_default_graph().get_tensor_by_name('fc' + '/kernel:0')
    sio.savemat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi_inv.mat', {'phi_inv':sess.run(phi_inv)})
    print(sess.run(phi_inv))
