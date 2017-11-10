
import tensorflow as tf
import numpy as np
from scipy import io as sio
import os
from tensorflow.contrib.data import Dataset, Iterator

blockSize = 16
measurement_rate = 0.25

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__)) + '/dataset/training_dataset.mat')

training_measurements = f['measurements']
training_measurements = np.squeeze(training_measurements).transpose([1,0])
training_patches = f['patches_vec']
training_patches = np.squeeze(training_patches).transpose([1,0])

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__)) + '/dataset/validation_dataset.mat')

validation_measurements = f['measurements']
validation_measurements = np.squeeze(validation_measurements).transpose([1,0])
validation_patches = f['patches_vec']
validation_patches = np.squeeze(validation_patches).transpose([1,0])

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__)) + '/dataset/phi.mat')
phi = f['phi']

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__)) + '/dataset/psi.mat')
psi = f['psi']

measurements_placeholder = tf.placeholder(training_measurements.dtype, [None, np.ceil(measurement_rate * (blockSize ** 2))], name='measurements')
patches_placeholder = tf.placeholder(training_patches.dtype, [None, blockSize ** 2], name='patches')

training_dataset = Dataset.from_tensor_slices((measurements_placeholder, patches_placeholder)).batch(100)
validation_dataset = Dataset.from_tensor_slices((measurements_placeholder, patches_placeholder)).batch(1000)

nEpochs = 5

iterator = Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

next_measurement, next_patch = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

def build_phi_inv(measurement):
    patch = tf.layers.dense(measurement, blockSize ** 2, use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float64), name='fc_phi_inv')
    
    coeffs_est = tf.matmul(tf.convert_to_tensor(psi, dtype=tf.float64), patch, transpose_a=True, transpose_b=True, name='coeffs')

    return patch

def build_loss(patch, patch_est):
    loss = tf.losses.mean_squared_error(labels=patch, predictions=patch_est)
    return loss

patch_est = build_phi_inv(next_measurement)
coeffs = tf.matmul(next_patch, psi, transpose_a=False, transpose_b=True);

loss = build_loss(next_patch, patch_est)

training_op = tf.train.AdamOptimizer().minimize(loss) 

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)

    for i in range(nEpochs):
        print('EPOCH NO. %d' % i)

        sess.run(training_init_op, feed_dict={measurements_placeholder:training_measurements, patches_placeholder:training_patches})

        while True:
            try:
                sess.run(training_op)

            except tf.errors.OutOfRangeError:
                break

        sess.run(validation_init_op, feed_dict={measurements_placeholder:validation_measurements, patches_placeholder:validation_patches})

        while True:
            try:
                print('VALIDATION LOSS: %f' % sess.run(loss))

            except tf.errors.OutOfRangeError:
                break


    tf.train.Saver().save(sess, os.path.dirname(os.path.abspath(__file__)) + '/dataset/tmp/model_sparse_linear_mapping.ckpt', latest_filename='checkpoint_sparse_linear_mapping')