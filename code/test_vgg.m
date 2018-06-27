%% landmark train
clear all; clc;
caffe.reset_all();
% figure;
mOx = [];mOxCr = [];mOxLw = [];mOxB = [];
mPa = [];mPaCr = [];mPaLw = [];mPaB = [];
iter_all = [];

addpath('util');

% net_style = 'vgg-16';
% net_style = 'ft_weight_sum_b3_720_embdding_landmark_0.0005';
net_style = 'ft_b4_sgd_L2_0.001_5';
gpu_device = 3;
start_epoch = 1;
epoch_step = 2;
model_fold =  '/home1/qcz/qcz_pro/building_train/code/model/new_experiment/contrastiveloss/concat/with_pca/L2/conv5';
% model_fold = '/home1/qcz/qcz_pro/metric/triplet/vgg16';
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

im_fn_test = cellfun(@(x) [im_folder_test, x, '.jpg'], gnd_test.imlist, 'un', 0);
im_fn_train = cellfun(@(x) [im_folder_train, x, '.jpg'], gnd_train.imlist, 'un', 0);

load('/home1/qcz/DataSet/building_imgs/imdb.mat');
cids = imdb.imglist;
qidxs = [imdb.train_pair(:,1)',imdb.val_pair(:,1)'];
pidxs = [imdb.train_pair(:,2)',imdb.val_pair(:,2)'];

img_fn_test = im_fn_test;
img_fn_train = im_fn_test;
img_cids = cids;


for i = 1:length(im_fn_test)
    img_fn_test{i} = imread(im_fn_test{i});
    disp(i);
end
for i = 1:length(im_fn_train)
    img_fn_train{i} = imread(im_fn_train{i});
    disp(i);
end
for i = 1:length(cids)
    img_cids{i} = imread(cids{i});
    disp(i);
end


for iter = start_epoch:epoch_step:35
    
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_device);


%%k

model.model_file = fullfile(model_fold,'test.prototxt');
model.weights_file = fullfile(model_fold,['epoch_',num2str(iter),'.caffemodel']);
if ~exist(model.weights_file,'file') continue;end

%%
model.maxDim = 1024;

net = caffe.Net(model.model_file,model.weights_file,'test');


% parameters of the method
use_gpu	 				= 1;		% use GPU to get CNN responses
% extract features
fprintf('Extracting features\n');
tic
vecs_test = cellfun(@(x) vecpostproc(mac((x), net, model)), img_fn_test, 'un', 0);
toc
tic
vecs_train = cellfun(@(x) vecpostproc(mac((x), net, model)), img_fn_train, 'un', 0);
toc
vecs_whiten = {};
model.maxDim = 0;
tic
for i=1:numel(cids)
    vecs_whiten{i} = vecpostproc(mac(img_cids{i},net,model));
end
toc
model.maxDim = 1024;

vecs_whiten = cell2mat(vecs_whiten);
fprintf('>> whitening: Learning...\n');
Lw = whitenlearn(vecs_whiten, qidxs, pidxs);
save('./data/Lw.mat','Lw','-v7.3');
[~,eigvecP,eigvalP,XmP] = yael_pca(single(cell2mat(vecs_train')));
[~,eigvecO,eigvalO,XmO] = yael_pca(single(cell2mat(vecs_test')));
[~,eigvecB,eigvalB,XmB] = yael_pca(single(vecs_whiten));
%% test oxford5k and paris6k

% apply PCA-whitening
vecs_test = cell2mat(vecs_test');
vecs_train = cell2mat(vecs_train');
save('./data/vecs.mat','vecs_test','vecs_train','-v7.3');
vecsOxLw = whitenapply(vecs_test, Lw.m, Lw.P);
vecsPaLw = whitenapply(vecs_train, Lw.m, Lw.P);
vecsOxCr = apply_whiten(vecs_test,XmP,eigvecP,eigvalP);
vecsPaCr = apply_whiten(vecs_train,XmO,eigvecO,eigvalO);
vecsOxB = apply_whiten(vecs_test,XmB,eigvecB,eigvalB);
vecsPaB = apply_whiten(vecs_train,XmB,eigvecB,eigvalB);
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
qvecsOxLw = whitenapply(qvecsOx, Lw.m, Lw.P);
qvecsOxCr = apply_whiten(qvecsOx,XmP,eigvecP,eigvalP);
qvecsOxB = apply_whiten(qvecsOx,XmB,eigvecB,eigvalB);

qvecsPa = cell2mat(qvecsPa);
qvecsPaLw = whitenapply(qvecsPa, Lw.m, Lw.P);
qvecsPaCr = apply_whiten(qvecsPa,XmO,eigvecO,eigvalO);
qvecsPaB = apply_whiten(qvecsPa,XmB,eigvecB,eigvalB);

% retrieval with inner product
[ranksOx,~] = yael_nn(vecs_test, -qvecsOx, size(vecs_test, 2), 16);
mapOx = compute_map (ranksOx, gnd_test.gnd);
[ranksOxCr,~] = yael_nn(vecsOxCr, -qvecsOxCr, size(vecsOxCr, 2), 16);
mapOxCr = compute_map (ranksOxCr, gnd_test.gnd);
[ranksOxLw,~] = yael_nn(vecsOxLw, -qvecsOxLw, size(vecsOxLw, 2), 16);
mapOxLw = compute_map (ranksOxLw, gnd_test.gnd);
[ranksOxB,~] = yael_nn(vecsOxB, -qvecsOxB, size(vecsOxB, 2), 16);
mapOxB = compute_map (ranksOxB, gnd_test.gnd);


[ranksPa,~] = yael_nn(vecs_train, -qvecsPa, size(vecs_train, 2), 16);
mapPa = compute_map (ranksPa, gnd_train.gnd);
[ranksPaCr,~] = yael_nn(vecsPaCr, -qvecsPaCr, size(vecsPaCr, 2), 16);
mapPaCr = compute_map (ranksPaCr, gnd_train.gnd);
[ranksPaLw,~] = yael_nn(vecsPaLw, -qvecsPaLw, size(vecsPaLw, 2), 16);
mapPaLw = compute_map (ranksPaLw, gnd_train.gnd);
[ranksPaB,~] = yael_nn(vecsPaB, -qvecsPaB, size(vecsPaB, 2), 16);
mapPaB = compute_map (ranksPaB, gnd_train.gnd);

model.maxDim = 1024;
fprintf('mAP, oxford5k = %.4f  %.4f  %.4f %.4f\nmAP, oxford5k = %.4f  %.4f  %.4f %.4f\n', mapOx,mapOxCr,mapOxLw,mapOxB,mapPa,mapPaCr,mapPaLw,mapPaB);

mOx = [mOx,mapOx];
mOxCr = [mOxCr,mapOxCr];
mOxLw = [mOxLw,mapOxLw];
mOxB = [mOxB,mapOxB];
mPa = [mPa,mapPa];
mPaCr = [mPaCr,mapPaCr];
mPaLw = [mPaLw,mapPaLw];
mPaB = [mPaB,mapPaB];

iter_all = [iter_all,iter];
switchFigure(1);
clf;
subplot(1,2,1);
title('oxford5k');
xlabel('epoch');ylabel('map');
plot(iter_all,mOx,'b--o',iter_all,mOxCr,'g--o',iter_all,mOxLw,'r--o',iter_all,mOxB,'k--o');
legend('raw','cross','lw','bow','Location','southeast');

subplot(1,2,2);
title('paris6k');
xlabel('epoch');ylabel('map');
plot(iter_all,mPa,'b--o',iter_all,mPaCr,'g--o',iter_all,mPaLw,'r--o',iter_all,mPaB,'k--o');
legend('raw','cross','lw','bow','Location','southeast');

suptitle(net_style);
drawnow;
saveas(gcf,saveFigName);
save(saveReName,'mOx','mOxCr','mOxLw','mOxB','mPa','mPaCr','mPaLw','mPaB','-v7.3');
end
