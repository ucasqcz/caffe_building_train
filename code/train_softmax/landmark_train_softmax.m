%% train landmark with softmax
clear all;clc;
addpath('../util');

%% caffe init
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(1);
model_fold = '../model';
pre_train_weight = fullfile(model_fold,'vgg-16', 'vgg16.caffemodel');
pre_train_mean = fullfile(model_fold,'vgg-16','mean_image.mat');
ft_model = fullfile('softmax','finetune_model_softmax','train.prototxt');
ft_solver = fullfile('softmax','finetune_model_softmax','solver.prototxt');
