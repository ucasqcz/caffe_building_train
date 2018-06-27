import numpy as np
import sys,os
import scipy.io as scio
import _init_Path
import caffe
from prepare import prepare_model
from compute_map import compute_map


gpu_device = 1
start_epoch = 1
epoch_step = 2
model_fold = '../code/model/new_experiment'
net_style = 'test'


epoch_num = 10;
model_fold = os.path.join(model_fold,net_style)
model = prepare_model(model_fold)
model['fea_name'] = 'avg_norm'
model['model_file'] = os.path.join(model_fold,'test')
model['weight_file'] = os.path.join(model_fold, "epoch_%d.caffemodel" % epoch_num)

caffe.set_mode_gpu()
caffe.set_device(start_epoch)
net = caffe.Net(model[''])

data_folder = '/home1/qcz/DataSet'
dataset_train,dataset_test = 'paris6k','oxford5k'
gnd_test = os.path.join(data_folder,dataset_test,'gnd_'+dataset_test+'.mat')
gnd_test = scio.loadmat(gnd_test)
qidx = gnd_test['qidx']
imlist = gnd_test['imlist']
gnd = gnd_test['gnd']
rank_path = '/home1/qcz/qcz_pro/building_train/code/ranks.mat'
ranks = scio.loadmat(rank_path)
ranks = ranks['ranksOxLw']
[map,aps] = compute_map(ranks,gnd)
gnd_train = os.path.join(data_folder,dataset_train,'gnd_'+dataset_train+'.mat')
gnd_train = scio.loadmat(gnd_train)

