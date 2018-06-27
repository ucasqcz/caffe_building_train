%% landmark train
clear all; clc;

% figure;
mOx = [];mOxCr = [];mOxLw = [];mOxB = [];
mPa = [];mPaCr = [];mPaLw = [];mPaB = [];
iter_all = [];

addpath('util');

% net_style = 'vgg-16';
net_style = 'ft_weight_sum_b4_720_embedding_landmark_0.002_iter_size/model_bk';
gpu_device = 3;
start_epoch = 31;
epoch_step = 2;
model_fold = './model/new_experiment';
% model_fold = './model/new_experiment';
model_fold = fullfile(model_fold,net_style);
model = prepare_model(model_fold);
if(isempty(findstr(net_style,'sum')))
    model.fea_name = 'avg_norm';
else
    model.fea_name = 'avg_norm';
end
iter = 21;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_device);


%%k

model.model_file = fullfile(model_fold,'test.prototxt');
model.weights_file = fullfile(model_fold,['epoch_',num2str(iter),'.caffemodel']);


%%
model.maxDim = 1024;

net = caffe.Net(model.model_file,model.weights_file,'test');


% parameters of the method
use_gpu	 				= 1;		% use GPU to get CNN responses
% extract features
fprintf('Extracting features\n');
%% 100k fea
tic;
ex_fea_name = ['./data/ex_fea_',num2str(iter),'.mat'];
try
    load(ex_fea_name);
catch
    ex_fold = '/home1/qcz/DataSet/Oxford100k/images';
    ex_list = dir(ex_fold);ex_list = ex_list(3:end);
    ex_name = {ex_list.name};ex_path = cellfun(@(x) fullfile(ex_fold,x),ex_name,'un',0);
    vecs_extra = cellfun(@(x) vecpostproc(mac((x), net, model)), ex_path, 'un', 0);
    vecs_extra = cell2mat(vecs_extra');
    save(ex_fea_name,'vecs_extra','ex_path','-v7.3');
end
toc






