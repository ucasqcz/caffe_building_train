function batch = gen_batch_val(net,imdb,bthid,batchsize,neg_num)
    tuple_per_num = neg_num + 1;
    batchsize = batchsize * tuple_per_num;
    total_num = max(size(imdb.val_pair));
    st = mod(bthid*batchsize+1,total_num);
    en = mod((bthid+1) * batchsize,total_num);
    if(st == 0) st = total_num;end
    if(en == 0) en = total_num;end
    if(en<st)
        sub_pair = [imdb.val_pair(st:end,:);imdb.val_pair(1:en,:)];
    else
        sub_pair = imdb.val_pair(st:en,:);
    end
%     startidx = (bthid - 1)*batchsize +1;
%     endidx = bthid * batchsize;
%     sub_pair = imdb.val_pair(startidx:endidx,:);
    
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    data = zeros(input_size(1),input_size(2),3,batchsize * 2);
    label = ones(1,1,1,batchsize);
    for i = 1:batchsize
        idx_q = sub_pair(i,1);
        idx_p = sub_pair(i,2);
        im_q = prepare_blob_for_train(imdb.imglist{idx_q},net,imdb.mean_img);
        im_p = prepare_blob_for_train(imdb.imglist{idx_p},net,imdb.mean_img);
        data(:,:,:,i) = im_q;
        data(:,:,:,i + batchsize) = im_p;
    end
        %% four input
    label_extra = ones(1,1,1,batchsize);
    batch = {data,label,label_extra};
end