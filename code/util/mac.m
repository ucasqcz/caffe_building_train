function [x, X] = mac(I, net, model)
    [img,~] = prepare_blob_for_cnn(I,model);
    if ischar(I)
        fprintf('%s : dealing with img: %s \n',mfilename,I);
    end
    fea_name = model.fea_name;
    net.blobs('data').reshape([size(img,1),size(img,2),3,1]);
    net.reshape();
    net.forward({img});
    X = net.blobs(fea_name).get_data();
	x = mac_act(X);