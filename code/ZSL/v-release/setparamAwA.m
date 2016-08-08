function Opt = setparamAwA(datasetname, paramVersion, LoadParamFlag)
%this block used to add path ,set some params, and load the attribute data
%input:{datasetname,paramVersion,whether or not load data}
%output:parameters(include num-class,attribute data, attribute-dim)
fprintf(['Set common parameters for ',datasetname,' dataset....\n']);
Opt.path.root = '/'; % windows
Opt.path.code = [Opt.path.root, 'code/ZSL/v-release'];
Opt.path.support = [Opt.path.root, 'support/'];
Opt.path.liblinear = [Opt.path.support, 'liblinear-2.01/matlab'];
Opt.path.libsvm    = [Opt.path.support, 'libsvm-3.20/matlab'];
Opt.path.structsvm = [Opt.path.support, 'svm-struct-matlab-1.2'];
Opt.path.SLEP      = [Opt.path.support, 'SLEP_package_4.1'];
Opt.inputpath = [Opt.path.root, 'dataset/'];
Opt.outputpath = [Opt.path.root, 'results/'];
Opt.path.feature = [Opt.inputpath, datasetname, '/feature_mat/'];
Opt.path.wordvector = [Opt.inputpath, datasetname, '/wordvector_mat/'];

addpath(Opt.path.liblinear);
addpath(Opt.path.libsvm);
addpath(Opt.path.structsvm);
addpath(genpath(Opt.path.SLEP));

Opt.dataset = datasetname;
if LoadParamFlag == true
    load([Opt.outputpath, datasetname, '/AwA_param_v',num2str(paramVersion),'.mat']);
else
    renewSeenUnseen = false; %true;
    load([Opt.inputpath, datasetname, '/constants.mat']);  %include all information about data set and classes
    load([Opt.inputpath, datasetname, '/nperclass.mat']);
    Opt.v_fold = 1; %5, seperate train data into 5 folds, 4 fraction for train, 1 for validation.
    Opt.nperclass = nperclass;
    Opt.classes = classes;
    Opt.oldfeat = feat;
    Opt.oldnfeat = nfeat;
    Opt.oldtrainclasses_id = trainclasses_id;
    Opt.oldtestclasses_id = testclasses_id;
    if renewSeenUnseen == true
        idrnd = randperm(length(classes));
        Opt.trainclasses_id = idrnd(1:length(Opt.oldtrainclasses_id));
        Opt.testclasses_id = idrnd(length(Opt.oldtrainclasses_id)+1:end);
    else
        Opt.trainclasses_id = Opt.oldtrainclasses_id;
        Opt.testclasses_id = Opt.oldtestclasses_id;
    end   
    Opt.featname = 'vgg+Goog+res152_fc';
    Opt.featdim = 3024;            
    %KES: Knowledge Embedding Space for Transfer Learning (Zero-Shot Learning)
    load([Opt.path.wordvector, 'AwA_wcb500.mat']);    w_cbow500 = clsVecAwA;    
    Opt.KES.name = 'a_prob+w_cbow';
    Opt.KES.dim  = 585;
    pca_prob      = NormalizeTo_0_1(pca_prob);
    w_cbow500     = NormalizeTo_0_1(w_cbow500);
    Opt.KES.anchors = [pca_prob, w_cbow500];   
end
save([Opt.outputpath, datasetname, '/AwA_param_v',num2str(paramVersion),'.mat'], 'Opt');

function B = NormalizeTo_0_1(A)
v = max(A(:)) - min(A(:));
if v > 0 
    B = (A - min(A(:)))/v;
else
    B = A;
end