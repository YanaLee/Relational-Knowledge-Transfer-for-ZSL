clear; close all;% clc;
fprintf('\n Relational Knowledge Transfer for Zero-Shot Learning \n');
fprintf('-----------------------------------------------------------------------------------\n');
%%
%Describe the dataname, feature, and the prediction method
TIscript.type = 'F2K2C';
TIscript.datamattype = 'KRT';
TIscript.ESZSL.model = 1; 
TIscript.F2K2Cpredicator ='ESZSL';
TIscript.dataset = 'CUB';

%%
%add the useful path
%load the attributes and set some param
switch TIscript.dataset
	case 'AwA'
		TIscript.lparaspace = 0;        
		TIscript.gparaspace = 5.4;         
		TIscript.SC_degree = 0.5;
		TIscript.NPC = 1500;
		TIscript.SIGMA = 0.1; 
		Opt  = setparamAwA(TIscript.dataset, 1.0, false);
        %use the feature vgg_vd19_fc8+GoogLeNet, attritube a_prob+w_cbow
	case 'CUB'
		TIscript.lparaspace = 0.2;        
		TIscript.gparaspace = 4;         
		TIscript.SC_degree = 0.05;
		TIscript.NPC = 100;
		TIscript.SIGMA = 0.1; 
		Opt  = setparamCUB(TIscript.dataset, 1.0, false);
        %use the feature vgg_vd19_fc8+GoogLeNet1024s, attritube a_prob+w_skipgram
	case 'Dogs'
        TIscript.lparaspace = 0.1;        
		TIscript.gparaspace = 4.4;         
		TIscript.SC_degree = 0.3;
		TIscript.NPC = 50;
		TIscript.SIGMA = 0.1; 
		Opt  = setparamDogs(TIscript.dataset, 1.0, false);
        %use the feature vgg+Goog+res152_fc, attritube a_prob+w_skipgram
end
%%
%the main function
LoadDataFlag = false;
Opt.KRT.nAugpercls = TIscript.NPC;
Opt.KRT.lamda = TIscript.SC_degree;
Opt.KRT.sigma = TIscript.SIGMA;   % Gaussian noise variance
Opt.ESZSL.model = TIscript.ESZSL.model;
Opt.F2K2Cpredicator = TIscript.F2K2Cpredicator;
Data = generateKRTdata(Opt, LoadDataFlag);%get the data by KRT
ESZSL.lamda = 10^TIscript.lparaspace;
ESZSL.gamma = 10^TIscript.gparaspace;
F2K2C = f2k2c_ESZSL(Opt, Data, ESZSL.gamma, ESZSL.lamda);%classification



