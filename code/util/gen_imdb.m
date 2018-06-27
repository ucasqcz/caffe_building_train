function imdb = gen_imdb(dataset_fold)

    img_fold = fullfile(dataset_fold,'imgs');
    trainval_file = fullfile(dataset_fold,'ims_trainval.mat');
    imdb_file = fullfile(dataset_fold,'imdb.mat');

    data = load(trainval_file);
    clu = data.cluster;
    imglist = arrayfun(@(x,y) fullfile(img_fold,[num2str(x),'_',num2str(y),'.jpg']),1:numel(clu),clu,'un',0);
    im_total = zeros(362,362,3,'single');
    for i = 1:length(imglist)
        im = imread(imglist{i});
        im = imresize(im,[362,362],'bilinear');
        im = single(im);
        im_total = im_total + im;
        disp(i);
    end
    mean_img = im_total / length(imglist);
    
    imdb.mean_img = mean_img;
    imdb.dataset_fold = dataset_fold;
    imdb.src_file = trainval_file;
    imdb.img_fold = img_fold;
    imdb.imglist = imglist;
    imdb.trainset = find(data.set == 1);
    imdb.valset = find(data.set == 2);
    train = [data.train.qidxs',data.train.pidxs'];
    imdb.train_pair = train;
    val = [data.val.qidxs',data.val.pidxs'];
    imdb.val_pair = val;
    imdb.cluster = clu;

    
    save(imdb_file,'imdb','-v7.3');   
end