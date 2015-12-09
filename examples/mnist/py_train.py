# -*- coding: utf-8 -*-
#train model by py-interface
from __future__ import division
import numpy as np
import sys
caffe_root = '../../' 
sys.path.insert(0, caffe_root + 'python')
import caffe

#finetune: copy weight from a existing model
# we fintune from a trained model
base_weights = 'lenet_iter_10000.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('lenet_solver.prototxt')

#finetune, copy weight from a trained model. optional
solver.net.copy_from(base_weights)
net = solver.net
conv1_w = net.params['conv1'][0].data
#iterate 10 times 
solver.step(10)

#cause of a well trained model, loss would be small enough after 10 iters
print net.blobs['loss'].data
conv1_w_after_iter = net.params['conv1'][0].data

conv1_w_diff = (conv1_w_after_iter - conv1_w).sum()
#weight stay put
print 'sum of weight update:', conv1_w_diff

#now you can go on training
#solver.step(10000)

#or visualize the weights, modify weights to do net surgery 
#and what ever you want
