function Data = generateKRTdata(Opt, LoadDataFlag) %Knowlwdge Relationship Transfer
% This function first use the spase coding of K2C matrix to learn the relation between test classes and train
% classes; then generate several virtual points to augment the whole training dataset for the
% improvement of F2K(eg. feature-to-attribute) prediction accuracy.
if LoadDataFlag == true
    fprintf(['Step#2: load KRT train and test data for ',Opt.dataset,' dataset (',Opt.KES.name,')....\n']);
    load([Opt.outputpath, Opt.dataset, '/DataKRT_[', Opt.featname, ']=[',...
        Opt.KES.name, '].mat']);
else
    fprintf(['Generate KRT train and test data for ',Opt.dataset,' dataset (',Opt.KES.name,')....\n']);
    load([Opt.path.feature, Opt.dataset, '_', Opt.featname, '.mat']); % feature dataset files
    X = X';
    
    Y_kes = [];
    y_cls = [];
    for i=1:length(Opt.nperclass)
        y_cls = [y_cls; repmat(i,Opt.nperclass(i),1)];
        Y_kes = [Y_kes; repmat(Opt.KES.anchors(i,:),Opt.nperclass(i),1)];
    end
    Data.Y_kes = Y_kes;
    Data.y_cls = y_cls;
    
    TestData = [];
    TestLabel_K =[];
    TestLabel_c = [];
    TrainData = [];
    TrainDataMu = []; % mean vector of each training class;
    TrainLabel_K =[];
    TrainLabel_c = [];
    ValidationData = [];
    ValidationLabel_K =[];
    ValidationLabel_c = [];
    for j=1:length(Opt.trainclasses_id)
        idx = find(y_cls==Opt.trainclasses_id(j));
        TrainData = [TrainData; X(idx,:)];
        TrainDataMu = [TrainDataMu; mean(X(idx, :))];
        TrainLabel_K = [TrainLabel_K; Y_kes(idx,:)];
        TrainLabel_c = [TrainLabel_c; y_cls(idx)];
    end
    for j=1:length(Opt.testclasses_id)
        idx = find(y_cls==Opt.testclasses_id(j));
        TestData = [TestData; X(idx,:)];
        TestLabel_K = [TestLabel_K; Y_kes(idx,:)];
        TestLabel_c = [TestLabel_c; y_cls(idx)];
    end
    TestDataMu = mean(TestData);
    if Opt.v_fold == 1
        VSet_num = length(TrainLabel_K(:,1));
        VTrainSet_num = VSet_num;
        vidx = randperm(VSet_num)';
    else
        VSet_num = length(TrainLabel_K(:,1));
        VTrainSet_num = uint32(VSet_num*(1.0-1.0/v_fold));
        vidx = randperm(VSet_num)';
        ValidationData = TrainData(vidx(VTrainSet_num+1:end),:);
        ValidationLabel_K = TrainLabel_K(vidx(VTrainSet_num+1:end),:);
        ValidationLabel_c = TrainLabel_c(vidx(VTrainSet_num+1:end));
    end
    
    Data.TrainData = sparse(TrainData(vidx(1:VTrainSet_num),:));
    Data.TrainLabel_K = TrainLabel_K(vidx(1:VTrainSet_num),:);
    Data.TrainLabel_c = TrainLabel_c(vidx(1:VTrainSet_num));
    Data.ValidationData = sparse(ValidationData);
    Data.ValidationLabel_K = ValidationLabel_K;
    Data.ValidationLabel_c = ValidationLabel_c;
    Data.TestData = sparse(TestData);
    Data.TestLabel_K = TestLabel_K;
    Data.TestLabel_c = TestLabel_c;
    Data.trainclasses_id = Opt.trainclasses_id;
    Data.testclasses_id = Opt.testclasses_id;
    
    Ktr = double(Opt.KES.anchors(Opt.trainclasses_id,:));
    Kts = double(Opt.KES.anchors(Opt.testclasses_id,:));
    [AugData, AugLabel_K, AugLabel_c] = generateKRTAugData(Opt, Ktr, Kts, TrainDataMu, TestDataMu);
    N_aug = length(AugLabel_c);
    augidx = randperm(N_aug);
    
    if N_aug>0
        Data.AugData = sparse(AugData(augidx,:));
        Data.AugLabel_K = AugLabel_K(augidx,:);
        Data.AugLabel_c = AugLabel_c(augidx,:);
    else
        Data.AugData = [];
        Data.AugLabel_K = [];
        Data.AugLabel_c = [];
    end
    
    save([Opt.outputpath, Opt.dataset, '/DataKRT_[', Opt.featname, ']=[',...
        Opt.KES.name, '].mat'], 'Data', '-v7.3');
end

function [AugData, AugLabel_K, AugLabel_c] = generateKRTAugData(Opt, Ktr, Kts, TrainDataMu, TestDataMu)
% compute the sparse coding of each testing class;
sc_test = zeros(size(Ktr,1), size(Kts,1));
for k = 1 : size(Kts,1)
    [sc_test(:, k)] = LeastR(Ktr', Kts(k, :)', Opt.KRT.lamda);
end
nAugpercls = Opt.KRT.nAugpercls;
if nAugpercls<=0
    AugData = [];
    AugLabel_K = [];
    AugLabel_c = [];
    return;
end
% use the mean vector of each class and gaussian noise to augment
dataGen = [];
for i = 1 : size(Ktr,1)
    xx = repmat(TrainDataMu(i, :), nAugpercls, 1) + randn(nAugpercls, Opt.featdim) * Opt.KRT.sigma;
    dataGen = [dataGen xx];
end
% generate the vitual data
dataAug1 = [];
for j = 1 : nAugpercls
    x = reshape(dataGen(j, :), Opt.featdim, length(Opt.trainclasses_id));
    a = x*sc_test; % the vitual data 4096*10;
    b = a(:)';
    dataAug1 = [dataAug1; b]; % nAug*40960;
end
% tranform the dataAug1 into the standard form
for i = 1 : size(Kts,1) % distribute these 10 data into 10 test classes respectively
    dataAug((i-1)*nAugpercls+1:i*nAugpercls, :) = dataAug1(:, (i-1)*Opt.featdim+1:i*Opt.featdim);
end
% augment the training data
AugData = dataAug;
AugLabel_K = [];
AugLabel_c = [];
for i = 1 : length(Opt.testclasses_id)
    AugLabel_K = [AugLabel_K; repmat(Kts(i,:), nAugpercls, 1)];
    AugLabel_c = [AugLabel_c; repmat(Opt.testclasses_id(i), nAugpercls, 1)];
end

