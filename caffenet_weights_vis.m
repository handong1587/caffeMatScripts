clear
clc
close all

addpath('../matlab');
caffe.set_mode_cpu();
fprintf(['Caffe Version = ', caffe.version(),'\n']);

net=caffe.Net('../models/bvlc_reference_caffenet/deploy.prototxt','../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel','test');

fprintf('Load net done. Net Layers :');
net.layer_names

fprintf('Net blobs : ');
net.blob_names

conv1_layer=net.layer_vec(2);
blob1=conv1_layer.params(1);
w=blob1.get_data();

fprintf('Conv1 Weight Shape: ');
size(w)
visualize_weights(w,1);

conv2_layer=net.layer_vec(6);
blob2=conv2_layer.params(1);
w2=blob2.get_data();
fprintf('conv2 weight shape:');
size(w2)
visualize_weights(w2,1);
