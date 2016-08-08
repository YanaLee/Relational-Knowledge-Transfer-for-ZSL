%%% generate eye/rand features for each dataset
% close all; clear; clc;
function gen_eye_rand_VW()

fea_fold = 'Z:\wdh\DeepAL_V201508\dataset';

fea_data = {'AwA', 'CUB', 'Dogs'};

fea_indx = [ 1,     2,          3,        4];
fea_name = {'eye', 'eye_rand', 'eye300', 'rand300'};
fea_dime = [ 50,    50,         300,      300];

for d = 1:length(fea_data)
    dataset = fea_data{d};
    
    for f = 1:length(fea_name)
        comp_info = sprintf('Generating %s feature for %s', fea_name{f}, dataset);
        fprintf('===== %s ...\n', comp_info);
        
        % load sample data
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_goog1024.mat']);
        load(fea_file, 'VW');
        numcls = size(VW,1);
        
        % generate data
        switch f
            case 1
                VW = eye(numcls); % eye
            case 2
                VW = rand(numcls); % eye_rand
            case 3
                VW = eye(fea_dime(3)); % eye300
                ind = randperm(fea_dime(3));
                VW = VW(ind(1:numcls),:);
            case 4
                VW = rand(numcls, fea_dime(4)); % rand300
            otherwise
                error(['Wrong feature number: ' num2str(f)]);
        end
        VW = NormalizeTo_0_1(VW);
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{f} '.mat']);
        save(fea_file, 'VW');
    end
end

function B = NormalizeTo_0_1(A)
v = max(A(:)) - min(A(:));
if v > 0 
    B = (A - min(A(:)))/v;
else
    B = A;
end