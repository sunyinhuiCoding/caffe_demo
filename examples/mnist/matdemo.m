% MNIST Matlab demo
% matcaffe needed
clear
caffe_root = '../../';
addpath(genpath([caffe_root,'matlab/']));
%set to 0 to use cpu caffe
use_gpu = 1;

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
test_imgs_dir = '../../data/figure/';
model_dir = './';
%model_dir = '../../examples/finetune/';
net_model = [model_dir 'lenet.prototxt'];
net_weights = [model_dir 'lenet_iter_10000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download caffemodel before you run this demo,refer README.md');
end
% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
%get data
test_img = imread('test.png');
if(size(test_img,3) ~= 1)
   test_img = test_img(:,:,1); 
end
test_img = imresize(test_img, [28, 28]);
im_data = permute(test_img, [2, 1]);  % flip width and height
im_data = single(im_data);            % convert from uint8 to single
score = net.forward({im_data});
score = score{:};

%% get and visualize model weights
layer_names = net.layer_names();
disp(['layer names:']);
disp(layer_names);
%get layer index of 'conv1'
conv1_index = net.name2layer_index('conv1');
%access parameters of conv1
conv1_param = net.layer_vec(conv1_index); 
conv1_param = conv1_param.params;
conv1_w = conv1_param(1).get_data();
disp('size of conv1 kernel weights:');
disp(size(conv1_w));
conv1_b = conv1_param(2).get_data();
disp('size of conv1 kernel bias:');
disp(size(conv1_b));
% you can see conv1_w is a 5x5x1x20 matrix since the conv kernel size is
% 5x5, and input blob channel is 1, output numbers is 20

%access conv2
conv2_index = net.name2layer_index('conv2');
conv2_param = net.layer_vec(conv2_index); 
conv2_param = conv2_param.params;
conv2_w = conv2_param(1).get_data();
conv2_b = conv2_param(2).get_data();

% weights for visualization
w1 = exp(conv1_w); w1 = w1/max(w1(:));
w2 = exp(conv2_w); w2 = w2/max(w2(:));
fg1 = figure(1);
for i=1:20
   subplot(4,10,i);imshow(w1(:,:,i));
   title(['conv1']);
end
for i=21:40
   subplot(4,10,i);imshow(w2(:,:,i - 20)); 
   title(['conv2']);
end

%% access layer output
% output of conv/pool 1-2
conv1_blob = net.blob_vec(net.name2blob_index('conv1')).get_data();
pool1_blob = net.blob_vec(net.name2blob_index('pool2')).get_data();
conv2_blob = net.blob_vec(net.name2blob_index('conv2')).get_data();
pool2_blob = net.blob_vec(net.name2blob_index('pool2')).get_data();
%output of inner product layer
ip2_blob = net.blob_vec(net.name2blob_index('ip2')).get_data();
figure(3);
for i=1:20
    subplot(8, 10,i);
    imshow(conv1_blob(:,:,i));
    title(['conv1'])
end
for i=21:40
    subplot(8,10,i);
    imshow(pool1_blob(:,:,i - 20));
    title(['pool1'])
end
for i=41:60
    subplot(8, 10,i);
    imshow(conv2_blob(:,:,i - 40));
    title(['conv2'])
end
for i=61:80
    subplot(8,10,i);
    imshow(pool2_blob(:,:,i - 60));
    title(['pool2'])
end

%% do net surgery:modify layer parameters
% add a random noise on conv1 and conv2 layer weights
net.params('conv1', 1).set_data(net.params('conv1', 1).get_data() + randn(size(net.params('conv1', 1).get_data())));
net.params('conv2', 1).set_data(net.params('conv2', 1).get_data() + randn(size(net.params('conv2', 1).get_data())));

%infer again using new noised weights
score_after_noise = net.forward({im_data});
score_after_noise = score_after_noise{:};
ip2_blob_after_noise = net.blob_vec(net.name2blob_index('ip2')).get_data();

disp(['original output of inner product layer:']);
disp(ip2_blob);
disp(['output of inner product layer after noised weight:']);
disp(ip2_blob_after_noise);

disp(['original output of probability:']);
disp(score);
disp(['output of probability layer after noised weight:']);
disp(score_after_noise);

caffe.reset_all();