%% sample generate
fold = '/home1/qcz/DataSet/building_imgs';
imdbFile = fullfile(fold,'imdb.mat');
imgFold = fullfile(fold,'imgs');
imdb = load(imdbFile);
imdb = imdb.imdb;
train_file = fopen('train.txt','w+');
test_file = fopen('val.txt','w+');
train_ratio = 0.8;
cl_uq = unique(imdb.cluster);
cl_num = numel(cl_uq);
for i = 1:cl_num
    c = cl_uq(i);
    idx = find(imdb.cluster == c);
    imgs = imdb.imglist(idx);
    label = c - 1;
    num = numel(idx);
    train_num = floor(num*train_ratio);
    train_set = imgs(1:train_num);
    val_set = imgs(train_num:num);
    for id = 1:num
        if(id<=train_num)
            fprintf(train_file,'%s %d\n',imgs{id},label);
        else
            fprintf(test_file,'%s %d\n',imgs{id},label);
        end
    end
end

fclose(train_file);
fclose(test_file);





