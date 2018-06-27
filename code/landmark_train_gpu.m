%% landmark train
function landmark_train()

clear all; clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(2);

dataset_fold = '/home1/qcz/DataSet/building_imgs';
imdb_file = fullfile(dataset_fold,'imdb.mat');
try
    load(imdb_file);
catch
    imdb = gen_imdb(dataset_fold);
end
% caffe.init_log('logo_file.log');

log_file = fopen(['output_log_',datestr(now,0),'.txt'],'w');

model_fold = './model';
pre_train_weight = fullfile(model_fold,'vgg-16', 'vgg16.caffemodel');
pre_train_mean = fullfile(model_fold,'vgg-16','mean_image.mat');

ft_model = fullfile(model_fold,'finetune_model','train.prototxt');
ft_solver = fullfile(model_fold,'finetune_model','solver.prototxt');

solver = caffe.get_solver(ft_solver);
solver.net.copy_from(pre_train_weight);

iter_ = solver.iter();
max_iter = solver.max_iter();
fea_name = 'mac';
batchsize = 5;
mining_iter_ = 0;
neg_num = 5;
tuple_per_num = neg_num + 1;

fprintf('start hard mining---\n');
[train_tuples,val_tuples] = hard_mining(solver.net,imdb,fea_name,mining_iter_,batchsize*tuple_per_num*2,tuple_per_num);
fprintf('finish hard mining---\n');



mining_num_per_epoch = 1;
mining_per_iter_num = floor(length(train_tuples) / (mining_num_per_epoch * batchsize));
show_per_iter = 20;
val_iter_ = 0;

epoch = 0;

total_img_data = load(imdb.src_file);
total_img_data = total_img_data.data;


batch = gen_batch(solver.net,imdb,train_tuples,iter_,batchsize,total_img_data,tuple_per_num);
save('batch.mat','batch','-v7.3');
while(iter_ < max_iter)
    solver.net.set_phase('train');
	
    % epoch = floor(iter_ * batchsize / length(train_tuples));
	
   
	
	%batch_save_name = ['batch_',num2str(iter_),'.mat'];
	%save(batch_save_name,'batch','-v7.3');
	
    solver.net.set_input_data(batch);
    solver.step(1);
	
    train_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
    train_loss = train_loss.get_data();
    
    if(mod(iter_,mining_per_iter_num) == 0)
        mining_iter_ = floor(iter_ / mining_per_iter_num);
        [train_tuples,val_tuples] = hard_mining(solver.net,imdb,fea_name,mining_iter_,batchsize*tuple_per_num*2,tuple_per_num);
    end
    %% show error and acc
    if(mod(iter_,show_per_iter) == 0)
	%{
		mac_q = solver.net.blob_vec(solver.net.name2blob_index('mac_q'));
		mac_q = mac_q.get_data();mac_q = squeeze(mac_q);
		mac_p = solver.net.blob_vec(solver.net.name2blob_index('mac_p'));
		mac_p = mac_p.get_data();mac_p = squeeze(mac_p);
		label = solver.net.blob_vec(solver.net.name2blob_index('label'));
		label = label.get_data();label = squeeze(label);
		dis = sum((mac_q - mac_p).^2,1);
		
		
		for n = 1:numel(dis)/tuple_per_num
			disp(dis((n-1)*tuple_per_num +1:n*tuple_per_num));
		end
		
		dis = sqrt(dis);
		% disp(label');
		calc_loss = sum((dis.^2).*label');
		dis = 0.7 - dis;
		dis(dis<0) = 0;
		calc_loss = calc_loss + sum((dis.*(1-label')).^2);
		
		calc_loss = calc_loss / (2 * numel(dis));
		% fprintf('calc loss is : %f---\n',calc_loss);
		% fprintf(log_file,'calc loss is : %f---\n',calc_loss);
	%}	
        test_batch = gen_batch(solver.net,imdb,val_tuples,val_iter_,batchsize,total_img_data,tuple_per_num);
        solver.net.set_phase('test');
        solver.net.forward(test_batch);
        test_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
		test_loss = test_loss.get_data();
        val_iter_ = val_iter_ + 1;
        fprintf('iter %d --- train loss: %f test loss %f\n',iter_,train_loss,test_loss);
		fprintf(log_file,'iter %d --- train loss: %f test loss %f\n',iter_,train_loss,test_loss);
    end
	
	epoch_num = floor(iter_*batchsize/length(train_tuples));
	if epoch_num ~= epoch
		save_fold = '/data1/NLPRMNT/wanghongsong/qcz/ObjectRetrieval/ICCV/caffe_mac/model/finetune_model/';
		save_model_file = [save_fold,'model_',num2str(epoch_num),'.caffemodel'];
		solver.net.save(save_model_file);
		epoch = epoch_num;
	end
	
    iter_ = solver.iter(); 
end
end

function batch = gen_batch(net,imdb,train_tuples,iter_,batchsize,total_img_data,tuple_per_num)
    tuple_num = length(train_tuples);
    st = mod(iter_  * batchsize + 1,tuple_num);
	if(st == 0) st = tuple_num;end
    en = mod((iter_ + 1) * batchsize,tuple_num);
	if(en == 0) en = tuple_num;end
    if (en < st) 
		sub_train_tuples = [train_tuples(st:end,:);train_tuples(1:en,:)];
	else
		sub_train_tuples = train_tuples(st:en,:);
	end
	% fprintf('st: %d---en: %d \n',st,en);
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    data = zeros(input_size(1),input_size(2),3,batchsize * tuple_per_num * 2);
    label = zeros(1,1,1,batchsize * tuple_per_num);
    %% query
    for idx = 1:batchsize
        id = sub_train_tuples(idx,1);
        % im = prepare_blob_for_train(imdb.imglist{id},net,imdb.mean_img);
		im = prepare_blob_for_train(total_img_data{id},net,imdb.mean_img);
        for i = 1:tuple_per_num
            data(:,:,:,(idx - 1) * tuple_per_num + i) = im;
        end
    end
    %% pos + neg
    for idx = 1: batchsize
        for i = 2:(tuple_per_num + 1)
            id = sub_train_tuples(idx,i);
            % im = prepare_blob_for_train(imdb.imglist{id},net,imdb.mean_img); 
			im = prepare_blob_for_train(total_img_data{id},net,imdb.mean_img); 
            data_idx = batchsize * tuple_per_num + (idx - 1) * tuple_per_num + i - 1;
            data(:,:,:,data_idx) = im;
        end
    end
    %% label
    for idx = 1:batchsize
        for i = 1:tuple_per_num
            if(mod(i - 1,tuple_per_num) == 0)
                label(:,:,:,(idx - 1) * tuple_per_num + i) = 1;
            end
        end
    end
    batch = {data,label};
end
function imdb = gen_imdb(dataset_fold)

    img_fold = fullfile(dataset_fold,'imgs');
    trainval_file = fullfile(dataset_fold,'ims_trainval.mat');
    imdb_file = fullfile(dataset_fold,'imdb.mat');
	mean_file = fullfile(dataset_fold,'mean.mat');
	
	mean_img = load(mean_file);
	mean_img = mean_img.mean_img;
	
    vari = load(trainval_file);
	cluster = vari.cluster;
	train = vari.train;
	val = vari.val;
	set = vari.set;
	
    imglist = arrayfun(@(x,y) fullfile(img_fold,[num2str(x),'_',num2str(y),'.jpg']),1:numel(cluster),cluster,'un',0);
	%{
    im_total = zeros(362,362,3,'single');
    for i = 1:length(imglist)
        im = imread(imglist{i});
        im = imresize(im,[362,362],'bilinear');
        im = single(im);
        im_total = im_total + im;
        disp(i);
    end
    mean_img = im_total / length(imglist);
    %}
    imdb.mean_img = mean_img;
    imdb.dataset_fold = dataset_fold;
    imdb.src_file = trainval_file;
    imdb.img_fold = img_fold;
    imdb.imglist = imglist;
    imdb.trainset = find(set == 1);
    imdb.valset = find(set == 2);
    train = [train.qidxs',train.pidxs'];
    imdb.train_pair = train;
	imdb.cluster = cluster;
    val = [val.qidxs',val.pidxs'];
    imdb.val_pair = val;

    
    save(imdb_file,'imdb','-v7.3');   
end

function [train_tuples,val_tuples] = hard_mining(net,imdb,fea_name,mining_iter,batchsize,tuple_per_num)
    net.set_phase('test');
    trainset_path = imdb.imglist(imdb.trainset);
    trainset_cluster = imdb.cluster(imdb.trainset);
    
    valset_path = imdb.imglist(imdb.valset);
    valset_cluster = imdb.cluster(imdb.valset);
    
    [totalset_path,totalset_cluster] = extend_set(imdb.imglist,imdb.cluster,batchsize);
    %% hard mining according to the feature
	save_fold = 'mining_data';
	if ~exist(save_fold,'dir') mkdir(save_fold);end
    save_fea_name = ['total_fea_',fea_name,'_',num2str(mining_iter),'.mat'];
	save_fea_name = fullfile(save_fold,save_fea_name);
    try
        load(save_fea_name);
    catch
		% load the imgs
		fprintf('start load all the img---\n');
		d_st = clock;
		data = load(imdb.src_file);
		data = data.data;
		d_en = clock;
		fprintf('load img finished  time is : %f---\n',etime(d_en,d_st));
        total_fea = get_fea_cnn(totalset_path,net,imdb.mean_img,batchsize,fea_name,data);
        save(save_fea_name,'total_fea','-v7.3');
    end
    save_tuple_name = ['tuples_',num2str(mining_iter),'.mat'];
	save_tuple_name = fullfile(save_fold,save_tuple_name);
    try
        load(save_tuple_name);
    catch
        [train_tuples,val_tuples] = mining_tuples(imdb,total_fea,tuple_per_num);
        save(save_tuple_name,'train_tuples','val_tuples','-v7.3');
    end
%     [train_tuples,val_tuples] = mining_tuples(imdb,total_fea);
    %% randperm
    rand_order_train = randperm(size(train_tuples,1));
    rand_order_val = randperm(size(val_tuples,1));
    train_tuples = train_tuples(rand_order_train,:);
    val_tuples = val_tuples(rand_order_val,:);
    
end

function [train_tuples,val_tuples] = mining_tuples(imdb,total_fea,tuple_per_num)
    train_tuples = gen_tuples(imdb,total_fea,'train',tuple_per_num);
    val_tuples = gen_tuples(imdb,total_fea,'val',tuple_per_num);
    %% show tuples
	fprintf('saving mining tuples--\n');
    save_fold = 'train_tuples';
    show_tuples(imdb,train_tuples,save_fold);
    save_fold = 'val_tuples';
    show_tuples(imdb,val_tuples,save_fold);
    
end
function tuples = gen_tuples(imdb,total_fea,set_name,tuple_per_num)
    if(strcmp(set_name,'train'))
        fea = total_fea(imdb.trainset,:);
        cluster = imdb.cluster(imdb.trainset);
        pair = imdb.train_pair;
        id_set = imdb.trainset;
    elseif(strcmp(set_name,'val'))
        fea = total_fea(imdb.valset,:);
        cluster = imdb.cluster(imdb.valset);
        pair = imdb.val_pair;
        id_set = imdb.valset;
    end
    tuples = [];
    for i = 1:size(pair,1)
        q = pair(i,1);
        p = pair(i,2);
        q_fea = total_fea(q,:);
        q_cluster = imdb.cluster(q);
        max_cluster = max(cluster);
        dis_total = bsxfun(@minus,fea,q_fea);
        dis_total = sum(dis_total.^2,2);
        % order : dis + id + cluster
        min_per_cluster = [];    
        for c = 1:max_cluster
            if (c == q_cluster)
                continue;
            end
            memb = find(cluster == c);
            if isempty(memb)
                continue;
            end
            c_id = id_set(memb);
            dis = dis_total(memb,:);
            [min_dis,idx] = min(dis);
            min_member_id = c_id(idx);
            tmp_min_per_cluster = [min_dis,min_member_id,c];
            min_per_cluster = [min_per_cluster;tmp_min_per_cluster];
        end
        %% sort and keep top 5
        [re,neg_idx] = sort(min_per_cluster(:,1));
        neg_sample = min_per_cluster(neg_idx(1:(tuple_per_num - 1)),2);
        tmp_tuple = [q,p,neg_sample'];
        tuples = [tuples;tmp_tuple];
		if(mod(i,300) == 0)
			fprintf('%s : %d th tuple--\n',set_name,i);
		end
    end
    
end
function show_tuples(imdb,tuples,save_fold)
    rand_order = randperm(size(tuples,1));
    sub_num = min(size(tuples,1),100);
    sub_tuples = tuples(rand_order(1:sub_num),:);
    
    if ~exist(save_fold,'dir') mkdir(save_fold);end
    for i = 1:size(sub_tuples,1)
        q = sub_tuples(i,1);
        p = sub_tuples(i,2);
        neg = sub_tuples(i,3:end);
        sub_fold = fullfile(save_fold,num2str(q));
        if ~exist(sub_fold,'dir') mkdir(sub_fold);end
        q_src_path = imdb.imglist{q};
        q_dst_path = fullfile(sub_fold,['query_',num2str(imdb.cluster(q)),'.jpg']);
        p_src_path = imdb.imglist{p};
        p_dst_path = fullfile(sub_fold,['postive_',num2str(imdb.cluster(p)),'.jpg']);
        copyfile(q_src_path,q_dst_path,'f');
        copyfile(p_src_path,p_dst_path,'f');
        
        for n = 1:size(neg,2)
            n_src_path = imdb.imglist{neg(n)};
            n_dst_path = fullfile(sub_fold,['negative_',num2str(n),'_',num2str(imdb.cluster(neg(n))),'.jpg']);
            copyfile(n_src_path,n_dst_path,'f');
        end
    end
        
end
function fea = get_fea_cnn(setpath,net,mean_img,batchsize,fea_name,data)
	net.set_phase('test');
    num = length(setpath);
    fea = [];
    for i = 1:(num / batchsize)
		io_st = clock;
        st = (i - 1)*batchsize + 1;
        en = i*batchsize;
        for idx = 1:batchsize
            % img_path = setpath{idx + st -1 };
            % tmp_data = prepare_blob_for_train(img_path,net,mean_img);
			tmp_idx = idx + st - 1;
			if(tmp_idx >= length(data))
				tmp_idx = length(data);
			end
			
			tmp_data = prepare_blob_for_train(data{tmp_idx},net,mean_img);
            blob(:,:,:,idx) = tmp_data;
        end
		io_en = clock;
		io_time = etime(io_en,io_st);
		
		fea_st = clock;
        label = zeros(1,1,1,batchsize / 2);
        net.forward({blob,label});
        fea_batch = net.blobs(fea_name).get_data();
        fea_batch = squeeze(fea_batch);
        fea = [fea;fea_batch'];
		fea_en = clock;
		fea_time = etime(fea_en,fea_st);
		if(mod(i,50) == 0)
			fprintf('%s : calc %d th batch img fea---io time is : %f  fea time is : %f---\n',mfilename,i,io_time,fea_time);
		end
    end
end
function [path,cluster] = extend_set(path,cluster,batchsize)
    % extend the set for exactly divisible by  batchsize
    num = length(path);
    extra_num = batchsize - mod(num,batchsize);
    for k = 1:extra_num
        extra_path{k} = path{num};
        extra_cluster(k) = cluster(num);
    end
    path = [path,extra_path];
    cluster = [cluster,extra_cluster];
end
function data = prepare_blob_for_train(im,net,mean_img)
    if ischar(im)
        im = imread(im);
    end
    if size(im,3) == 1
        im = repmat(im,[1,1,3]);
    end
    
    im = im(:,:,[3,2,1]);
    im = permute(im,[2,1,3]);
    im = single(im);
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    mean_img = imresize(mean_img,input_size);
    im = imresize(im,input_size);
    data = im - mean_img;
end