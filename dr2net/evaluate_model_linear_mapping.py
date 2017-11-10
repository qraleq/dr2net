
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import color, io, filters
import os

image = color.rgb2gray(io.imread(os.path.dirname(os.path.abspath(__file__))+'/dataset/images/testing/house.tif'))

f = sio.loadmat(os.path.dirname(os.path.abspath(__file__))+'/dataset/phi.mat')
phi = f['phi']

# img_plt = plt.imshow(phi)
# plt.show()

reconstruction = np.zeros(image.shape)
reconstruction_resnet = np.zeros(image.shape)
reconstruction_linear = np.zeros(image.shape)

imH, imW = image.shape[:2]
blockSize = 16
measurement_rate = 0.25

def psnr(image1, image2):
    mse = np.mean(np.square(image1-image2))
    PIXEL_MAX = 1
    return 10*np.log10(PIXEL_MAX**2/mse)


with tf.Session() as sess:
    dir(tf.contrib)
    saver = tf.train.import_meta_graph(os.path.dirname(os.path.abspath(__file__))+ '/dataset/tmp/model_linear_mapping.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(os.path.abspath(__file__))+'/dataset/tmp/', latest_filename='checkpoint_linear_mapping'))

    graph = tf.get_default_graph()

    measurement_placeholder = graph.get_tensor_by_name('IteratorGetNext:0')
    patch_placeholder = graph.get_tensor_by_name('IteratorGetNext:1')

    patch_est_linear_placeholder = graph.get_tensor_by_name('fc/MatMul:0')    
    
    test_patches = np.zeros((1,256))


    for r in np.arange(imH - blockSize + 1, step=blockSize):
        for c in np.arange(imW - blockSize + 1, step=blockSize):
        
            measurement = np.dot(phi, np.reshape(image[r:r + blockSize, c:c + blockSize], [-1, 1]))
            measurement = measurement.transpose()

            patch_est_linear = sess.run(patch_est_linear_placeholder, feed_dict={measurement_placeholder:measurement, patch_placeholder:test_patches})
        
            reconstruction[r:r + blockSize, c:c + blockSize] = np.reshape(patch_est_linear, [16, 16])
        
    print(psnr(image, reconstruction))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(reconstruction)
    plt.show()

