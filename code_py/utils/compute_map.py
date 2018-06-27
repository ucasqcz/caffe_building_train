import numpy as np
import sys,os
import scipy.io as scio

def score_ap_from_ranks1(ranks,n):
    nimgranks = ranks.shape[0]
    ap = 0
    recall_step = 1.0 / n
    precision_0 = 0.0
    for i in range(1,nimgranks+ 1):
        rank = ranks[i - 1]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = (i - 1) / rank
        precision_1 = i / (rank + 1)
        ap = ap + (precision_0 + precision_1)*recall_step / 2
    return ap


def compute_map(ranks,gnd):
    map = 0
    nq = gnd.shape[0]
    aps = np.zeros((nq,1))
    for i in range(0,nq):
        rank = ranks[:,i]
        qgnd = np.array(gnd[i][0][0][0]);qgndj = np.array(gnd[i][0][1][0])
        sgnd = np.intersect1d(qgnd,rank);jgnd = np.intersect1d(qgndj,rank)
        pos = np.zeros((sgnd.shape[0],1));junk = np.zeros((jgnd.shape[0],1))
        for j in range(0,sgnd.shape[0]):
            pos[j] = np.where(rank == sgnd[j])
        for k in range(0,jgnd.shape[0]):
            junk[k] = np.where(rank == jgnd[k])
        pos = np.sort(pos,axis=0);junk = np.sort(junk,axis=0)
        k = 0;ij = 0;
        if junk.shape[0]:
            ip = 0
            while(ip < pos.shape[0]):
                while(ij < junk.shape[0] and pos[ip] > junk[ij]):
                    k = k + 1;ij = ij + 1
                pos[ip] = pos[ip] - k; ip = ip + 1
        ap = score_ap_from_ranks1(pos,qgnd.shape[0])
        map = map + ap
        aps[i] = ap
    map = map / nq
    return map,aps
'''
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
map,aps = compute_map(ranks,gnd)
'''