# -*- coding: utf-8 -*-
#using pycaffe interface to test and access model parameters
import numpy as np
import Image
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

img_file_name = 'test.png'
img = Image.open(img_file_name)
img = np.array(img, dtype=np.float32)
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
model_root = './'
net = caffe.Net(model_root+'lenet.prototxt','lenet_iter_10000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
# run net and take argmax for prediction
net.forward()

#get layer outputs
prob = net.blobs['prob'].data[0][:]
print 'output probabilities:\n',prob, '\n'

#access model paramaters
layer_names = [i for i in net._layer_names]
print 'layer names:\n', layer_names, '\n'
conv_layers = [i for i in net._layer_names if 'conv' in i]
print 'shape of conv filters:'
for i in conv_layers:
    print '  ', i, ':', net.params[i][0].data.shape
conv1_w = net.params['conv1'][0].data

#access blobs(layer outputs)
blob_names = [i for i in net._blob_names]
print 'blob names:\n', blob_names, '\n'
print 'shape of blobs:'
for i in blob_names:
    print '  ', i, ':', net.blobs[i].data.shape

#do net surgery:modify weights:
#set conv1 layer weights and bias to zero:
print '\ndo net surgery: set conv1 weights and bias to zero...'
net.params['conv1'][0].data[:] = 0
net.params['conv1'][1].data[:] = 0

#forward again:
print 'forward again with new weights, get new output probabilities...'
net.forward()
print '\nnew probabilities:', net.blobs['prob'].data[0][:], '\n'