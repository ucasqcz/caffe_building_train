function [imdb] = hard_mining(net,imdb_in,bthid,opts)
% bk mining_iter,batchsize,tuple_per_num

% bthid : 起始 batch 
% mining_per_batch_num: 每次mining 的batch 数目
% batchsize

    net.set_phase('test');
    trainset_path = imdb_in.imglist(imdb_in.trainset);
    trainset_cluster = imdb_in.cluster(imdb_in.trainset);
    valset_path = imdb_in.imglist(imdb_in.valset);
    valset_cluster = imdb_in.cluster(imdb_in.valset);
   
    opts.dA.flip = 0;
    
    %% trainset feature
    num_per_batch = 0;
    if strcmp(opts.training_method,'triplet')
        num_per_batch = 3;
    else
        num_per_batch = (opts.neg_num + 1) * 2;
    end
    if strcmp(opts.mode,'train')
        [total_path,total_cluster] = extend_set(trainset_path,trainset_cluster,opts.batchsize*num_per_batch);
    elseif strcmp(opts.mode,'val')
        [total_path,total_cluster] = extend_set(valset_path, valset_cluster,opts.batchsize*num_per_batch);
    end
    %% hard mining according to the feature
    
    total_fea = get_fea_cnn(total_path,net,imdb_in.mean_img,opts);
    
    [imdb] = mining_tuples(imdb_in,total_fea,bthid,opts);
end