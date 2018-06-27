function [path,cluster] = extend_set(path,cluster,batchsize)
    % extend the set for exactly divisible by  batchsize
    num = length(path);
    extra_num = batchsize - mod(num,batchsize);
    for k = 1:extra_num
        extra_path{k} = path{num};
        extra_cluster(k) = cluster(num);
    end
    path = [path,extra_path];
    cluster = [cluster,extra_cluster];
end