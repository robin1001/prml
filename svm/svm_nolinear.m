load data2.mat
load groups2.mat

%linear
svmStruct = svmtrain(data2,groups2,'showplot',true);

%kernel
figure;
sigma = [0.01 0.1 0.2 0.5 1.0];
for i = 1: length(sigma)
    subplot(1, length(sigma), i);
    title(['sigma = ' num2str(sigma(i))]);
    svmStruct = svmtrain(data2,groups2, 'kernel_function', 'rbf', 'rbf_sigma', sigma(i), 'showplot',true);
end

%we can see the result is apparently different, so it is very important to
%select optimal sigma in rbf svm