% function [imdb] = mining_tuples(imdb_in,total_fea,bthid,opts)
%     imdb = gen_tuples(imdb_in,total_fea,bthid,opts);
% end
function imdb = gen_tuples(imdb_in,total_fea,bthid,opts)

    mode = opts.mode;
    mining_per_batch_num = opts.mining_per_batch_num;
    batchsize = opts.batchsize;
    neg_num = opts.neg_num;
    
    cluster = imdb_in.cluster(imdb_in.([mode,'set']));
    fea = total_fea(1:numel(cluster),:);
    pair = imdb_in.([mode,'_pair']);
    id_set = imdb_in.([mode,'set']);
    
    tuples = [];
    if strcmp(mode,'train')
        startidx = (bthid -1)*batchsize + 1;
        endidx = min((bthid + mining_per_batch_num -1) *batchsize,size(pair,1));
    elseif strcmp(mode,'val')
        startidx = 1;
        endidx = size(pair,1);
    end
    total_dis = total_fea*fea';
    progressbar(0);
    for i = startidx:endidx
        q = pair(i,1);q = find(id_set == q);q = q(1);
%         p = pair(i,2);p = find(id_set == p);p = p(1);
%         q_fea = total_fea(q,:);
        q_cluster =cluster(q);
        
        noncluidx = find(cluster~=q_cluster);
%         [~,sortidx] = sort(fea(noncluidx,:)*q_fea','descend');
        [~,sortidx] = sort(total_dis(q,noncluidx),'descend');
        noncluidx = noncluidx(sortidx);
        
        [~,cidx] = unique(cluster(noncluidx),'stable');
        negidx = noncluidx(cidx);
        neg_id = id_set(negidx); % negative ID
        
        imdb_in.negidx(i,:) = neg_id(1:neg_num);
%         if(mod(i - startidx,300) == 0)
%             fprintf('%s : %d th tuple--\n',mode,i);
%         end
        progressbar((i - startidx) / (endidx - startidx));
    end
    imdb = imdb_in;
end
function show_tuples(imdb,tuples,save_fold)
    rand_order = randperm(size(tuples,1));
    sub_num = min(size(tuples,1),100);
    sub_tuples = tuples(rand_order(1:sub_num),:);
    
    if ~exist(save_fold,'dir') mkdir(save_fold);end
    for i = 1:size(sub_tuples,1)
        q = sub_tuples(i,1);
        p = sub_tuples(i,2);
        neg = sub_tuples(i,3:end);
        sub_fold = fullfile(save_fold,num2str(q));
        if ~exist(sub_fold,'dir') mkdir(sub_fold);end
        q_src_path = imdb.imglist{q};
        q_dst_path = fullfile(sub_fold,['query_',num2str(imdb.cluster(q)),'.jpg']);
        p_src_path = imdb.imglist{p};
        p_dst_path = fullfile(sub_fold,['postive_',num2str(imdb.cluster(p)),'.jpg']);
        copyfile(q_src_path,q_dst_path,'f');
        copyfile(p_src_path,p_dst_path,'f');
        
        for n = 1:size(neg,2)
            n_src_path = imdb.imglist{neg(n)};
            n_dst_path = fullfile(sub_fold,['negative_',num2str(n),'_',num2str(imdb.cluster(neg(n))),'.jpg']);
            copyfile(n_src_path,n_dst_path,'f');
        end
    end
        
end