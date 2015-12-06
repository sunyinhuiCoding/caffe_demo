#!/usr/bin/env sh
FINETUNE=0
if [ FINETUNE ];then
    ../../build/tools/caffe train --solver=lenet_solver.prototxt
else
	../../build/tools/caffe train --solver=lenet_solver.prototxt --weights=lenet_iter_10000.caffemodel
fi