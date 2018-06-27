%% test vgg qe%% landmark train
clear all; clc;

% figure;
mOx = [];mOxCr = [];mOxLw = [];mOxB = [];
mPa = [];mPaCr = [];mPaLw = [];mPaB = [];
iter_all = [];

addpath('util');

% net_style = 'vgg-16';
net_style = 'ft_weight_sum_b4_720_embedding_landmark_0.002_iter_size/model_bk';
gpu_device = 2;
start_epoch = 21;
iter = start_epoch;
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

% saveFigName = fullfile(model_fold,'map_ox_pa.jpg');
saveFigName = fullfile(model_fold,['map_',net_style,'.jpg']);
saveReName = fullfile(model_fold,['map_',num2str(start_epoch),'.mat']);
%% load dataset
data_folder = '/home1/qcz/DataSet'; % oxford5k/ and paris6k/ should be in here

dataset_train				= 'paris6k';    % dataset to learn the PCA-whitening on
dataset_test 				= 'oxford5k';     % dataset to evaluate on 

% config files for Oxford and Paris datasets 
gnd_test = load(fullfile(data_folder, dataset_test, ['gnd_',dataset_test,'.mat']));
gnd_train = load(fullfile(data_folder, dataset_train, ['gnd_',dataset_train,'.mat']));

% image files are expected under each dataset's folder
im_folder_test = fullfile(data_folder,dataset_test,'images/');
im_folder_train = fullfile(data_folder,dataset_train,'images/');

    
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
% apply PCA-whitening
load('./data/Lw.mat');
load('./data/vecs.mat');
load('./data/ex_fea_21.mat');
vecs_extra = reshape(vecs_extra,512,[]);
% vecs_test = [vecs_test,vecs_extra];
% vecs_train = [vecs_train,vecs_extra];
dim = 512;
mapOx = [];mapPa = [];
qe = 10;
for qe = 1:20;
vecsOxLw = whitenapply(vecs_test, Lw.m, Lw.P,dim);
vecsPaLw = whitenapply(vecs_train, Lw.m, Lw.P,dim);
%%  process query images
fprintf('Process query images\n');
qimlistOx = {gnd_test.imlist{gnd_test.qidx}};
qimOx = arrayfun(@(x) crop_qim([im_folder_test, qimlistOx{x}, '.jpg'], gnd_test.gnd(x).bbx,model.maxDim), 1:numel(gnd_test.qidx), 'un', 0);
model.maxDim = 0;
qvecsOx = cellfun(@(x) vecpostproc(mac(x, net, model)), qimOx, 'un', 0);

qimlistPa = {gnd_train.imlist{gnd_train.qidx}};
model.maxDim = 1024;
qimPa = arrayfun(@(x) crop_qim([im_folder_train, qimlistPa{x}, '.jpg'], gnd_train.gnd(x).bbx,model.maxDim), 1:numel(gnd_train.qidx), 'un', 0);
model.maxDim = 0;
qvecsPa = cellfun(@(x) vecpostproc(mac(x, net, model)), qimPa, 'un', 0);
% apply PCA-whitening on query vectors
qvecsOx = cell2mat(qvecsOx);
qvecsOxLw = whitenapply(qvecsOx, Lw.m, Lw.P,dim);

qvecsPa = cell2mat(qvecsPa);
qvecsPaLw = whitenapply(qvecsPa, Lw.m, Lw.P,dim);

% retrieval with inner product
[ranksOxLw,~] = yael_nn(vecsOxLw, -qvecsOxLw, size(vecsOxLw, 2), 16);
if qe~=0
    for i = 1:size(qvecsOxLw,2)
        qvecsOxLw(:,i) = qvecsOxLw(:,i) + sum(vecsOxLw(:,ranksOxLw(1:qe,i)),2);
        qvecsOxLw(:,i) = qvecsOxLw(:,i) /(qe + 1);
        qvecsOxLw(:,i) = vecpostproc(qvecsOxLw(:,i));
    end
end
[ranksOxLw,~] = yael_nn(vecsOxLw, -qvecsOxLw, size(vecsOxLw, 2), 16);
save('ranks.mat','ranksOxLw');
mapOxLw = compute_map (ranksOxLw, gnd_test.gnd);
mapOx = [mapOx,mapOxLw];
[ranksPaLw,~] = yael_nn(vecsPaLw, -qvecsPaLw, size(vecsPaLw, 2), 16);
if qe~=0
    for i = 1:size(qvecsPaLw,2)
        qvecsPaLw(:,i) = qvecsPaLw(:,i) + sum(vecsPaLw(:,ranksPaLw(1:qe,i)),2);
        qvecsPlraLw(:,i) = qvecsPaLw(:,i) /(qe + 1);
        qvecsPaLw(:,i) = vecpostproc(qvecsPaLw(:,i));
    end
end
[ranksPaLw,~] = yael_nn(vecsPaLw, -qvecsPaLw, size(vecsPaLw, 2), 16);

mapPaLw = compute_map (ranksPaLw, gnd_train.gnd);
mapPa = [mapPa,mapPaLw];
model.maxDim = 1024;
end
figure;
plot(1:20,mapOx,'r--',1:20,mapPa,'b--');
