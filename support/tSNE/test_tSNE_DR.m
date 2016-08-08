%%% PLot t-SNE of different features for each dataset
%%% 2016-05-10, Yuetan Lin
% close all; clear; clc;

fea_fold = 'X:\wdh\DeepAL_V201508\dataset';

fea_data = {'AwA', 'CUB', 'Dogs'};
fea_mark = {'o', '+'};

fea_indx = [ 1,         2,        3,          4,          5,         6,           7,     8,          9,        10];
fea_name = {'vgg1000', 'res_fc', 'goog1024', 'att_cont', 'wv_g300', 'wv_wcb300', 'eye', 'eye_rand', 'eye300', 'rand300'};
fea_dime = [ 1000,      1000,     1024,       85,         300,       300,         50,    50,         300,      300];
fea_mapp = cell(length(fea_name), 1);

lambda = 1e-4;

for d = 1:length(fea_data)
    dataset = fea_data{d};
    
    fea_choi = [3 1; 3 4; 3 2; 3 5; 3 7; 3 8; 3 9; 3 10]; % goog1024 and g300
    for c = 1:size(fea_choi,1)
        if d==3 && c<3 % Dogs do not have vgg1000 and att_cont
            continue;
        end
        choice = fea_choi(c,:);
        comp_info = sprintf('DR_Comparing %s with %s of %s', fea_name{choice(1)}, fea_name{choice(2)}, dataset);
        fprintf('===== %s ...\n', comp_info);
        
        % load data
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(1)} '.mat']);
        load(fea_file, 'VW');
        X = VW;
        
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(2)} '.mat']);
        load(fea_file, 'VW');
        Y = VW;
        numcls = size(VW,1);
        
        % DR
        %W = (X'*X)^(-1)*X'*Y;
        W = (X'*X+lambda*eye(size(X,2)))^(-1)*X'*Y;
        X = X*W;
        
        % t-SNE - 1. per feature
%         fea_mapp{choice(1)} = tsne(X, []);
%         fea_mapp{choice(2)} = tsne(Y, []);
        % t-SNE - 2. together
        tmp_fea_mapp = tsne([X; Y], [1:numcls 1:numcls]);
        fea_mapp{choice(1)} = tmp_fea_mapp(1:numcls,:);
        fea_mapp{choice(2)} = tmp_fea_mapp(numcls+1:end,:);
        
        % show
        sam_show = numcls;
        figure(1);clf;
        hold on;
        ind = randperm(numcls);
        idx = ind(1:sam_show);
        gscatter(fea_mapp{choice(1)}(idx,1), fea_mapp{choice(1)}(idx,2), idx, [], fea_mark{1});
        gscatter(fea_mapp{choice(2)}(idx,1), fea_mapp{choice(2)}(idx,2), idx, [], fea_mark{2});
        for k = 1:sam_show
            x1 = fea_mapp{choice(1)}(idx(k),1);
            y1 = fea_mapp{choice(1)}(idx(k),2);
            x2 = fea_mapp{choice(2)}(idx(k),1);
            y2 = fea_mapp{choice(2)}(idx(k),2);
            line([x1 x2], [y1 y2], 'Color', 'k');
        end
        xlabel(['\fontsize{14}' strrep(fea_name{choice(1)}, '_', '\_') ' (' fea_mark{1} ')']);
        ylabel(['\fontsize{14}' strrep(fea_name{choice(2)}, '_', '\_') ' (' fea_mark{2} ')']);
        title(['\fontsize{18}' strrep(sprintf('%s (Err %e)', comp_info, norm(Y'-X','fro')/numcls), '_', '\_')]);
        drawnow;
        print('-dpng', sprintf('feature_tSNE_result/%s/%s.png', dataset, comp_info));
        hold off;
        %pause();
    end
end