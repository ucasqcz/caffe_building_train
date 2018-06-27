%% test script
clear all;clc;
caffe.reset_all();
% caffe.set_mode_gpu();
% caffe.set_device(3);
caffe.set_mode_cpu();
fold = '/home1/qcz/qcz_pro/building_train/code/model/new_experiment/contrastiveloss/concat/with_pca/L1/ft_b2_sgd_L1_0.001_5/';
weight_file = fullfile(fold,'epoch_0.caffemodel');
model_file = fullfile(fold,'test.prototxt');
net = caffe.Net(model_file,weight_file,'test');

fold = '/home1/qcz/DataSet/paris6k/images/';
filelist = dir(fold);filename = {filelist(3:end).name};
filepath = cellfun(@(x) fullfile(fold,x),filename,'un',0);
diff = 0;
for i = 1:10%length(filepath)
mean_img = [122.67891434,116.66876762,104.00698793];
mean_img = mean(mean_img);
im =  imread(filepath{i});
im = single(im) - mean_img;
im = im(:,:,[3,2,1]);
im = permute(im,[2,1,3]);
im = single(im);
% im = imresize(im,[724,724]);
net.blobs('data').reshape([size(im,1),size(im,2),3,1]);
net.reshape();
net.forward({im});
conv5_3 = net.blobs('conv5_3').get_data();
% avg = net.blobs('mac').get_data();
% avg_off = mean(mean(conv5_3,1),2);
sw_0 = net.blobs('sw_0').get_data();
sw_norm_0 = net.blobs('sw_norm_0').get_data();
sw_multi_0 = net.blobs('sw_multi_0').get_data();
sw_result_0 = net.blobs('sw_result_0').get_data();
avg_0 = net.blobs('avg_0').get_data();
avg_off_0 = mean(mean(sw_result_0,1),2);

sw_1 = net.blobs('sw_1').get_data();
sw_norm_1 = net.blobs('sw_norm_1').get_data();
sw_multi_1 = net.blobs('sw_multi_1').get_data();
sw_result_1 = net.blobs('sw_result_1').get_data();
avg_1 = net.blobs('avg_1').get_data();
avg_off_1 = mean(mean(sw_result_1,1),2);

% avg_off = max(max(sw_result,[],1),[],2);
diff = diff + norm(squeeze(avg_0-avg_off_0));
avg_norm = net.blobs('avg_norm').get_data();
disp(i);
end
% diff = diff / length(filepath)
diff = diff /10