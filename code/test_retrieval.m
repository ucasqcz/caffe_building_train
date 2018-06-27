% Code to evaluate (not train) the methods presented in the paper
% F. Radenovic, G. Tolias, O. Chum, CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples, ECCV 2016

data_folder = '/datasets/'; % oxford5k/ and paris6k/ should be in here

dataset_test				= 'oxford5k';    % dataset to evaluate on (oxford5k or paris6k)

% config files for Oxford and Paris datasets
gnd_test = load([data_folder, dataset_test, '/gnd_', dataset_test, '.mat']);  % gnd_oxford5k and gnd_paris6k files provided

% image files are expected under each dataset's folder
im_folder_test = [data_folder, dataset_test, '/jpg/'];

% parameters of the method
use_rmac 				= 0;  	% use R-MAC, otherwise use MAC
use_gpu                 = 1;    % use GPU, otherwise use CPU

% choose pre-trained CNN model
modelfn = 'siaMAC_alex.mat'; whitenMACfn = 'Lw_MAC_alex.mat'; whitenRMACfn = 'Lw_RMAC_alex.mat'; % use fine-tuned AlexNet
% modelfn = 'siaMAC_vgg.mat'; whitenMACfn = 'Lw_MAC_vgg.mat'; whitenRMACfn = 'Lw_RMAC_vgg.mat';	% use fine-tuned VGG

load(modelfn);
net = dagnn.DagNN.loadobj(net) ;
if use_gpu
	gpuDevice(1);		% select GPU device to be used
	net.move('gpu');
end

im_fn_test = cellfun(@(x) [im_folder_test, x, '.jpg'], gnd_test.imlist, 'un', 0);

% extract features
fprintf('Extracting features. This can take a while...\n');
if ~use_rmac
	vecs = cellfun(@(x) mac(x, net), im_fn_test, 'un', 0);
else
	vecs = cellfun(@(x) rmac(x, net), im_fn_test, 'un', 0);
end

% apply learned whitening
fprintf('Applying learned whitening\n');
if ~use_rmac
	load(whitenMACfn);
else
	load(whitenRMACfn);
end
vecs = cellfun(@(x) apply_Lw (x, m, P), vecs, 'un', 0);

% process query images
fprintf('Process query images\n');
qimlist = {gnd_test.imlist{gnd_test.qidx}};
qim = arrayfun(@(x) crop_qim([im_folder_test, qimlist{x}, '.jpg'], gnd_test.gnd(x).bbx), 1:numel(gnd_test.qidx), 'un', 0);
if ~use_rmac
	qvecs = cellfun(@(x) mac(x, net), qim, 'un', 0);
else
	qvecs = cellfun(@(x) rmac(x, net), qim, 'un', 0);
end
% apply learned whitening on query vectors
qvecs = cellfun(@(x) apply_Lw (x, m, P), qvecs, 'un', 0);

fprintf('Retrieval\n');

% final database vectors and query vectors
vecs = cell2mat(vecs');
qvecs = cell2mat(qvecs);

% retrieval with inner product
sim = vecs'*qvecs;
[sim, ranks] = sort(sim, 'descend');
map = compute_map (ranks, gnd_test.gnd);
fprintf('mAP = %.4f\n', map);
