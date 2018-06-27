function fea = get_fea_cnn(setpath,net,mean_img,opts)

    if strcmp(opts.training_method,'triplet')
        batchsize = opts.batchsize * 3;
    else
        batchsize = opts.batchsize*(opts.neg_num + 1) * 2;
    end
    
    net.set_phase('test');
    num = length(setpath);
    fea = [];
    progressbar(0);
    for i = 1:(floor(num / batchsize))
        st = (i - 1)*batchsize + 1;
        en = i*batchsize;
        for idx = 1:batchsize
            img_path = setpath{idx + st -1 };
            data = prepare_blob_for_train(img_path,net,mean_img,opts);
            blob(:,:,:,idx) = data;
        end
        label = zeros(1,1,1,batchsize / 2);
        if strcmp(opts.training_method,'triplet')
            net.forward({blob});
        else
            net.forward({blob,label,label});
        end

        fea_batch = net.blobs(opts.fea_name).get_data();
        fea_batch = squeeze(fea_batch);
        fea = [fea;fea_batch'];
%         if(mod(i,200) == 0)
%             fprintf('%s : calc %d th batch img fea---\n',mfilename,i);
%         end
        progressbar(i / floor(num / batchsize));
    end
end