clear
clc
close all

addpath('matlab');
caffe.set_mode_cpu();
fprintf(['caffe version = ', caffe.version(),'\n']);

net=caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt','models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel','test');

fprintf('Load net done. Net layers :');
net.layer_names

fprintf('Net blobs :');
net.blob_names

fprintf('Now preparing data...\n');
im=imread('examples/images/cat.jpg');
figure; imshow(im);title('Original Image');
d=load('matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data=d.mean_data;
IMAGE_DIM=256;
CROPPED_DIM=227;

im_data=im(:,:,[3,2,1]);
im_data=permute(im_data,[2,1,3]);
im_data=single(im_data);
im_data=imresize(im_data,[IMAGE_DIM,IMAGE_DIM],'bilinear');
im_data=im_data-mean_data;  
im=imresize(im_data,[CROPPED_DIM,CROPPED_DIM],'bilinear');
km=cat(4,im,im,im,im,im);
pm=cat(4,km,km);
input_data={pm};

scores=net.forward(input_data);
scores=scores{1};
scores=mean(scores,2);

[~,maxlabel]=max(scores);

maxlabel
figure; plot(scores)

fm_data=net.blob_vec(1);
d1=fm_data.get_data();
fprintf('Data Size= ');
size(d1);
visualize_feature_maps(d1,1);

fm_conv1=net.blob_vec(2);
f1=fm_conv1.get_data();
fprintf('Feature map conv1 size= ');
size(f1);
visualize_feature_maps(f1,1);


