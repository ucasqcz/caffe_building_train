function batch = gen_batch(net,imdb,bthid,opts)
%     tuple_num = length(train_tuples);
%     st = mod(iter_  * batchsize + 1,tuple_num);
%     en = mod((iter_ + 1) * batchsize,tuple_num);
%     if(st == 0) st = tuple_num;end
%     if(en == 0) en = tuple_num;end
%     if(en<st)
%         sub_train_tuples = [train_tuples(st:end,:);train_tuples(1:en,:)];
%     else
%         sub_train_tuples = train_tuples(st:en,:);
%     end
    
    if(opts.dA.flip && strcmp(opts.mode,'train'))
        opts.dA.flip = 1;
    else
        opts.dA.flip = 0;
    end
    batchsize = opts.batchsize;
    neg_num = opts.neg_num;
    mode = opts.mode;

    startidx = (bthid - 1)*batchsize +1;
    endidx = bthid * batchsize;
    sub_tuples = [imdb.([mode,'_pair'])(startidx:endidx,:),imdb.negidx(startidx:endidx,:)];
    
    tuple_per_num = neg_num + 1;
    
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    data = zeros(input_size(1),input_size(2),3,batchsize * tuple_per_num * 2);
    label = zeros(1,1,1,batchsize *  tuple_per_num );
    label_extra = zeros(1,1,1,batchsize * tuple_per_num);
    %% query
    for idx = 1:batchsize
        id = sub_tuples(idx,1);
        im = prepare_blob_for_train(imdb.imglist{id},net,imdb.mean_img,opts);
        for i = 1:tuple_per_num
            data_q_idx = (idx - 1) * tuple_per_num + i;
            data(:,:,:,data_q_idx) = im;
            % for softmax
            label(:,:,:,data_q_idx) = imdb.cluster(id) - 1;
        end
    end

    %% pos + neg
    for idx = 1: batchsize
        for i = 2:(tuple_per_num + 1)
            id = sub_tuples(idx,i);
            im = prepare_blob_for_train(imdb.imglist{id},net,imdb.mean_img,opts); 
            data_idx = batchsize * tuple_per_num + (idx - 1) * tuple_per_num + i - 1;
            label_pn_idx = (idx - 1) * tuple_per_num + i - 1;
            data(:,:,:,data_idx) = im;
            label_extra(:,:,:,label_pn_idx) = imdb.cluster(id) - 1;
        end
    end
%     %% label
%     for idx = 1:batchsize
%         for i = 1:tuple_per_num
%             if(mod(i - 1,tuple_per_num) == 0)
%                 label(:,:,:,(idx - 1) * tuple_per_num + i) = 1;
%             end
%         end
%     end
%     %% four input
%     label_extra = ones(1,1,1,batchsize * tuple_per_num);
    batch = {data,label,label_extra};
end