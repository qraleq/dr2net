
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import color, io, filters
import os

image = color.rgb2gray(io.imread(os.path.dirname(os.path.abspath(__file__))+'/dataset/images/testing/001.jpg'))

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__))+'/dataset/phi.mat')
phi = f['phi']

reconstruction = np.zeros(image.shape)

imH, imW = image.shape[:2]

blockSize = 16
measurement_rate = 0.25

def psnr(image1, image2):
    mse = np.mean(np.square(image1-image2))
    PIXEL_MAX = 1
    return 10*np.log10(PIXEL_MAX**2/mse)


with tf.Session() as sess:
    dir(tf.contrib)
    saver = tf.train.import_meta_graph(os.path.dirname(os.path.abspath(__file__))+ '/dataset/tmp/model_ae.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(os.path.abspath(__file__))+'/dataset/tmp/', latest_filename='checkpoint_ae'))

    graph = tf.get_default_graph()

    measurements_placeholder = graph.get_tensor_by_name('measurements:0')
    patches_placeholder = graph.get_tensor_by_name('patches:0')

    measurement_est_placeholder = graph.get_tensor_by_name('fc_phi/MatMul:0')    
    patch_est_placeholder = graph.get_tensor_by_name('fc_phi_inv/MatMul:0')    
    
    init_op = graph.get_operation_by_name('MakeIterator')

    for r in np.arange(imH - blockSize + 1, step=blockSize):
        for c in np.arange(imW - blockSize + 1, step=blockSize):
            
            patch = np.reshape(image[r:r + blockSize, c:c + blockSize], [-1, 1])

            measurement = np.dot(phi, patch)
            measurement = measurement.transpose()

            sess.run(init_op, feed_dict={measurements_placeholder:measurement, patches_placeholder:patch.transpose()})

            patch_est = sess.run(patch_est_placeholder)

            reconstruction[r:r + blockSize, c:c + blockSize] = np.reshape(patch_est, [16, 16])

    print(psnr(image, reconstruction))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(reconstruction)
    plt.show()
    
    
