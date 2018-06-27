function data = prepare_blob_for_train(im,net,mean_img,opts)
    if ischar(im)
        im = imread(im);
    end
    if opts.dA.flip
        if(randi([0,1],1) == 1)
                im = flip(im,2);
        end
    end
    if size(im,3) == 1
        im = repmat(im,[1,1,3]);
    end
   
    input_size = net.blobs('data').shape();
    input_size = input_size(1:2);
    im = imresize(im,input_size);
    im = single(im);
%     mean_img = [122.67891434,116.66876762,104.00698793];
% %     mean_img = reshape(mean_img,[1,1,3]);
% %     mean_img = repmat(mean_img,input_size(1),input_size(2));
%     mean_img = mean(mean_img);
%     im = single(im) - mean_img;
    im(:,:,1) = im(:,:,1) - 122.678;
    im(:,:,2) = im(:,:,2) - 116.668;
    im(:,:,3) = im(:,:,3) - 104.006;
    
%     mean_img = imresize(mean_img,input_size);
%     im = single(im) - mean_img;


    
    im = im(:,:,[3,2,1]);
    im = permute(im,[2,1,3]);
    im = single(im);
    data = im;
end
