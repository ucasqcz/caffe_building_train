%% landmark train
clear all; clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);

opts.training_method = 'contrastive';
addpath('util');

dataset_fold = '/home1/qcz/DataSet/building_imgs';
imdb_file = fullfile(dataset_fold,'imdb.mat');
try
    load(imdb_file);
catch
    imdb = gen_imdb(dataset_fold);
end
% caffe.init_log('logo_file.log');
model_fold = './model/new_experiment/DiversityLoss';
fine_tune_type = 'ft_weight_sum_b4_720_embedding_landmark_0.001_iter_size_adam';
start_epoch = 1;
if(isempty(findstr(fine_tune_type,'sum')))
    opts.fea_name = 'mac_norm';
else
    opts.fea_name = 'avg_norm';
end
opts.batchsize = 5;
opts.neg_num = 2;
opts.mining_num_per_epoch = 2;

pre_train_weight = fullfile(model_fold,fine_tune_type, ['epoch_',num2str(start_epoch - 1),'.caffemodel']);
pre_train_mean = fullfile(model_fold,fine_tune_type,'mean_image.mat');

ft_model = fullfile(model_fold,fine_tune_type,'train.prototxt');
ft_solver = fullfile(model_fold,fine_tune_type,'solver.prototxt');

solver = caffe.get_solver(ft_solver);
solver.net.copy_from(pre_train_weight);

max_iter = solver.max_iter();

opts.tuple_per_num = opts.neg_num + 1;
opts.batch_num_per_epoch = floor(size(imdb.train_pair,1) / opts.batchsize);

opts.mining_per_batch_num = ceil(opts.batch_num_per_epoch / opts.mining_num_per_epoch);
opts.mode = 'train';
if(isempty(findstr(fine_tune_type,'da')))
    opts.dA.flip = 0;
else
    opts.dA.flip = 1;
end

show_per_iter = 50; 
iter_ =  0;
mining_iter_ = 0;

max_epoch_num = 50;

figure(1);
t_loss = [];
t_s_loss = [];
v_loss = [];
v_s_loss = [];

for epoch_num =start_epoch: max_epoch_num
    
    tic;
    rng(epoch_num);
    
    %% shuffer data
    order = randperm(size(imdb.train_pair,1));
    imdb.train_pair = imdb.train_pair(order,:);
    order = randperm(size(imdb.val_pair,1));
    imdb.val_pair = imdb.val_pair(order,:);
    
   
    imdb.negidx = zeros(size(imdb.train_pair,1),opts.neg_num,'single');
    
    mean_train_loss = 0;
    mean_train_s_loss = 0;
    opts.mode = 'train';
    
%     load('imdb.mat');
    for bthid = 1:opts.batch_num_per_epoch
        if(mod(bthid-1,opts.mining_per_batch_num) == 0)
%             [imdb] = hard_mining(solver.net,imdb,fea_name,bthid,mining_per_batch_num,batchsize,neg_num,'train');
            [imdb] = hard_mining(solver.net,imdb,bthid,opts);
        end     
        batch = gen_batch(solver.net,imdb,bthid,opts);
%         save('batch_test.mat','batch','-v7.3');
%         batch = load('batch_test.mat');
%         batch = batch.batch;
        solver.net.set_phase('train');
        solver.net.set_input_data(batch);
        solver.net.reshape();
        solver.step(1);
        iter_ = iter_ + 1;
        %% visual
        sw_norm_0 = solver.net.blob_vec(solver.net.name2blob_index('sw_0'));
        sw_norm_0_diff = sw_norm_0.get_data();
        sw_norm_0_diff = sw_norm_0_diff(:,:,:,1);sw_norm_0_diff = squeeze(sw_norm_0_diff);
        sw_norm_1 = solver.net.blob_vec(solver.net.name2blob_index('sw_1'));
        sw_norm_1_diff = sw_norm_1.get_data();
        sw_norm_1_diff = sw_norm_1_diff(:,:,:,1);sw_norm_1_diff = squeeze(sw_norm_1_diff);
        
        
        train_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
        train_loss = train_loss.get_data();
        mean_train_loss = (mean_train_loss*(bthid -1) + train_loss) / bthid ;
        
        train_s_loss = solver.net.blob_vec(solver.net.name2blob_index('diversity_loss'));
        train_s_loss = train_s_loss.get_data();
        mean_train_s_loss = (mean_train_s_loss*(bthid -1) + train_s_loss) / bthid ;
        
        if(mod(iter_ - 1,show_per_iter) == 0)
            fprintf('iter %d --- train loss: %f  diversity_loss: %f\n',iter_,mean_train_loss,mean_train_s_loss);
%             save_net = [num2str(iter_),'.caffemodel'];
%             solver.net.save(save_net);
        end
    end
    
    %% val loss
    fprintf('val one epoch\n');
    val_num = ceil(size(imdb.val_pair,1) / opts.batchsize / ((opts.neg_num + 1)));
    imdb.negidx = single.empty;
    opts.mode = 'val';
    [imdb] = hard_mining(solver.net,imdb,0,opts);
    mean_test_loss = 0;
    mean_test_s_loss = 0;
    
    for val_iter = 1:val_num
        test_batch = gen_batch(solver.net,imdb,val_iter,opts);
        solver.net.set_phase('test');
        solver.net.forward(test_batch);
        test_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
        test_loss = test_loss.get_data();
        mean_test_loss = (mean_test_loss*(val_iter -1) + test_loss) / val_iter;
        
        test_s_loss = solver.net.blob_vec(solver.net.name2blob_index('diversity_loss'));
        test_s_loss = test_s_loss.get_data();
        mean_test_s_loss = (mean_test_s_loss*(val_iter -1) + test_s_loss) / val_iter;
        
    end
    model_save = fullfile(model_fold,fine_tune_type,['epoch_',num2str(epoch_num),'.caffemodel']);
    solver.net.save(model_save);
    
    v_loss = [v_loss,mean_test_loss];
    v_s_loss = [v_s_loss,mean_test_s_loss];
    t_loss = [t_loss,mean_train_loss];
    t_s_loss = [t_s_loss,mean_train_s_loss];
    switchFigure(1);
    clf;
    subplot(1,4,1);
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,t_loss,'g-o');
    legend('train_contra');
    subplot(1,4,2);
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,t_s_loss,'r-o');
    legend('train_diversity');
    subplot(1,4,3);
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,v_loss,'g-o');
    legend('val_contra');
    subplot(1,4,4);
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,v_s_loss,'r-o');
    legend('val_diversity');
    
%     subplot(1,2,2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
%     xlabel('iter');ylabel('loss');
%     plot(start_epoch:epoch_num,v_loss,'g-o',start_epoch:epoch_num,v_s_loss,'r-o');
%     legend('val_contra','val_softmax');
    suptitle(fine_tune_type);
    
    drawnow;
    toc;
    saveas(gcf, fullfile(model_fold,fine_tune_type,[fine_tune_type,'.jpg']));
end