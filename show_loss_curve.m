clear
clc
close all

train_log_file = './examples/cifar10/cifar.log';
train_interval=100;
test_interval=500;

[~,string_output]=dos(['cat ', train_log_file, ' | grep ''Train net output #0''|awk ''{print $11}''']);
train_loss=str2num(string_output);
n=1:length(train_loss);
idx_train=(n-1)*train_interval;

[~,string_output]=dos(['cat ', train_log_file, ' | grep ''Test net output #1''|awk ''{print $11}''']);
test_loss=str2num(string_output);
m=1:length(test_loss);
idx_test=(m-1)*test_interval;
figure;plot(idx_train,train_loss);
hold on;
plot(idx_test,test_loss,'r');

grid on;
legend('Train Loss','Test Loss');
xlabel('iterations');
ylabel('loss');
title('Train & Test Loss Curve');

liu=1;
