function batch = gen_batch_triplet(net,imdb,bthid,opts)
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
    
    tuple_per_num = 3;
    
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    data = zeros(input_size(1),input_size(2),3,batchsize * tuple_per_num);
%     label = zeros(1,1,1,batchsize *  tuple_per_num );
    img_idx = sub_tuples(:);
    for i = 1:numel(img_idx)
        id = img_idx(i);
        im = prepare_blob_for_train(imdb.imglist{id},net,imdb.mean_img,opts);
        data(:,:,:,i) = im;
    end
    batch = {data};
end