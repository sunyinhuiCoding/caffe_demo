##caffe demos
几个caffe Matlab和python接口使用的demo。没有什么学术价值，仅仅是让大家熟悉一下caffe的matlab和python接口的使用。
###Matlab demo：
* net forward(infering)，load网络参数进行测试
* net parameter visualization，可视化网络参数(卷基层filter的weight)
* net surgery：人工改变网络的参数

###python demo
* training：使用python接口训练网络
* net forwad(infering)，load网络以及参数，进行测试
* net surgery

###how to use
* 使用make编译**caffe**(`make caffe`)以及**matcaffe**(`make matcaffe`),**pycaffe**(`make pycaffe`).
* 编译和caffe依赖库配置请参考<https://github.com/SHUCV/caffe_demo/wiki>
* 如果你要自己训练model的话，需要：  
  1 下载mnist数据集，执行`data/mnist/get_mnist.sh`完成数据下载；  
  2 将数据转换为`lmdb`格式，执行`examples/mnist/creat_mnist.sh`；  
  3 训练，执行`examples/mnist/train_lenet.sh`    
  
  **注意，以上1-2步的shell脚本必须在caffe根目录下执行！，3步的脚本在脚本目录执行**   

* 如果不想自己训练模型，可以[下载](http://7xocv2.dl1.z0.glb.clouddn.com/lenet_iter_10000.caffemodel)我训练好的model到`examples/mnist/`目录。
* 在`examples/mnist/`目录下执行matlab脚本`matdemo.m`，使用matlab接口进行模型预测，参数可视化以及参数调整。
* 在`examples/mnist/`目录下执行python脚本`pydemo.py`，使用python接口进行模型预测，参数调整。

  




___  
[zhaok]
