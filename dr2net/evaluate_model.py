
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import color, io, filters
#import dr2net_fc_At

image = color.rgb2gray(io.imread('Y:/Projects/Python Projects/dr2net/dr2net/dataset/images/testing/bear.tif'))

f = sio.loadmat('Y:/Projects/Python Projects/dr2net/dr2net/dataset/phi.mat')
phi = f['phi']

# img_plt = plt.imshow(phi)
# plt.show()

reconstruction = np.zeros(image.shape)
reconstruction_resnet = np.zeros(image.shape)
reconstruction_linear = np.zeros(image.shape)

imH, imW = image.shape[:2]
blockSize = 16
measurement_rate = 0.25


with tf.Session() as sess:
    dir(tf.contrib)
    saver = tf.train.import_meta_graph('Y:/Projects/Python Projects/dr2net/dr2net/dataset/tmp/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Y:/Projects/Python Projects/dr2net/dr2net/dataset/tmp/', latest_filename=None))

    graph = tf.get_default_graph()

    measurement_placeholder = graph.get_tensor_by_name('IteratorGetNext:0')
    patch_placeholder = graph.get_tensor_by_name('IteratorGetNext:1')

    patch_est_linear_placeholder = graph.get_tensor_by_name('fc/MatMul:0')    
    
    patch_est_resnet_placeholder = graph.get_tensor_by_name('Reshape_1:0')
    
    test_patches = np.zeros((1,256))


    for r in np.arange(imH - blockSize + 1, step=blockSize):
        for c in np.arange(imW - blockSize + 1, step=blockSize):
        
            measurement = np.dot(phi, np.reshape(image[r:r + blockSize, c:c + blockSize], [-1, 1]))
            measurement = measurement.transpose()

            patch_est_linear, patch_est_resnet = sess.run([patch_est_linear_placeholder, patch_est_resnet_placeholder], feed_dict={measurement_placeholder:measurement, patch_placeholder:test_patches})
        
            reconstruction[r:r + blockSize, c:c + blockSize] = np.reshape(patch_est_linear + patch_est_resnet, [16, 16])
            reconstruction_resnet[r:r + blockSize, c:c + blockSize] = np.reshape(patch_est_resnet, [16, 16])
            reconstruction_linear[r:r + blockSize, c:c + blockSize] = np.reshape(patch_est_linear, [16, 16])
        
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(reconstruction)
    plt.subplot(2,2,3)
    plt.imshow(reconstruction_linear)
    plt.subplot(2,2,4)
    plt.imshow(reconstruction_resnet)
    plt.show()

