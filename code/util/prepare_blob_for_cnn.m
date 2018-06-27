function [im,im_scale] = prepare_blob_for_cnn(im,model)
    if ischar(im)
        im = imread(im);
    end
    if size(im,3) == 1
        im = repmat(im,[1,1,3]);
    end
    % load mean file
%     mean_img = model.mean_img;
%     im = single(im) - repmat(mean_img,size(im,1),size(im,2));

	if model.maxDim == 0
		im_scale = 1;
	else
		max_size = model.maxDim;
		im_scale = max_size / max(size(im));
    end
    
    im = imresize(im,im_scale,'bicubic');
    im = single(im);
%     mean_img = mean(model.mean_img);
%     im = single(im)  - mean_img;
    im(:,:,1) = im(:,:,1) - 122.678;
    im(:,:,2) = im(:,:,2) - 116.668;
    im(:,:,3) = im(:,:,3) - 104.006;
    

    im = im(:,:,[3,2,1]);
    im = permute(im,[2,1,3]);
    im = single(im);
    
%     im = im - imresize(mean_img,[size(im,1),size(im,2)],'bilinear');
    
end