%% landmark train
clear all; clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(1);

addpath('util');

dataset_fold = '/home1/qcz/DataSet/building_imgs';
imdb_file = fullfile(dataset_fold,'imdb.mat');
try
    load(imdb_file);
catch
    imdb = gen_imdb(dataset_fold);
end
% caffe.init_log('logo_file.log');
model_fold = './model';
fine_tune_type = 'ft_5_0.4_da';
pre_train_weight = fullfile(model_fold,fine_tune_type, 'epoch_0.caffemodel');
pre_train_mean = fullfile(model_fold,fine_tune_type,'mean_image.mat');

ft_model = fullfile(model_fold,fine_tune_type,'train.prototxt');
ft_solver = fullfile(model_fold,fine_tune_type,'solver.prototxt');

solver = caffe.get_solver(ft_solver);
solver.net.copy_from(pre_train_weight);

max_iter = solver.max_iter();
opts.fea_name = 'mac_norm';
opts.batchsize = 3;
opts.neg_num = 4;
opts.tuple_per_num = opts.neg_num + 1;
opts.batch_num_per_epoch = floor(size(imdb.train_pair,1) / opts.batchsize);
opts.mining_num_per_epoch = 3;
opts.mining_per_batch_num = ceil(opts.batch_num_per_epoch / opts.mining_num_per_epoch);
opts.mode = 'train';
opts.dA.flip = 1;

show_per_iter = 100; 
iter_ =  0;
mining_iter_ = 0;

max_epoch_num = 50;

figure(1);
t_loss = [];
v_loss = [];
start_epoch = 1;
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
    opts.mode = 'train';
    
    for bthid = 1:opts.batch_num_per_epoch
        if(mod(bthid-1,opts.mining_per_batch_num) == 0)
%             [imdb] = hard_mining(solver.net,imdb,fea_name,bthid,mining_per_batch_num,batchsize,neg_num,'train');
            [imdb] = hard_mining(solver.net,imdb,bthid,opts);
        end       
%         batch = gen_batch(solver.net,imdb,bthid,batchsize,neg_num,'train');
        batch = gen_batch(solver.net,imdb,bthid,opts);
        
        solver.net.set_phase('train');
        solver.net.set_input_data(batch);
        solver.net.reshape();
        solver.step(1);
        iter_ = iter_ + 1;
        
        train_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
        train_loss = train_loss.get_data();
        mean_train_loss = (mean_train_loss*(bthid -1) + train_loss) / bthid ;
        
        if(mod(iter_ - 1,show_per_iter) == 0)
            fprintf('iter %d --- train loss: %f \n',iter_,mean_train_loss);
        end
    end
    
    %% val loss
    fprintf('val one epoch\n');
    val_num = ceil(size(imdb.val_pair,1) / opts.batchsize / ((opts.neg_num + 1)));
    imdb.negidx = single.empty;
    opts.mode = 'val';
    [imdb] = hard_mining(solver.net,imdb,0,opts);
    mean_test_loss = 0;
    
    for val_iter = 1:val_num
        test_batch = gen_batch(solver.net,imdb,val_iter,opts);
        solver.net.set_phase('test');
        solver.net.forward(test_batch);
        test_loss = solver.net.blob_vec(solver.net.name2blob_index('loss'));
        test_loss = test_loss.get_data();
        mean_test_loss = (mean_test_loss*(val_iter -1) + test_loss) / val_iter;
    end
    model_save = fullfile(model_fold,fine_tune_type,['epoch_',num2str(epoch_num),'.caffemodel']);
    solver.net.save(model_save);
    
    v_loss = [v_loss,mean_test_loss];
    t_loss = [t_loss,mean_train_loss];
    switchFigure(1);
    clf;
    subplot(1,2,1);
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,t_loss,'o-');
    legend('train');
    subplot(1,2,2);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    xlabel('iter');ylabel('loss');
    plot(start_epoch:epoch_num,v_loss,'o-');
    legend('val');
    drawnow;
    toc;
end
saveas(gcf, fullfile(model_fold,fine_tune_type,[fine_tune_type,'.jpg']));












