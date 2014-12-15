load data.mat
load groups.mat

%sample train, test
[train, test] = crossvalind('holdOut',groups, 0.2);
%train
svmStruct = svmtrain(data(train,:),groups(train),'showplot',true);
%test
classes = svmclassify(svmStruct,data(test,:),'showplot',true);
cp = classperf(groups);
classperf(cp,classes,test);
disp(cp.CorrectRate);