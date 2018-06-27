%% load dataset
clear all;clc

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
tic;
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
toc