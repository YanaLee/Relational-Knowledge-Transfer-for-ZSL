%%% From Feature X to Knowledge K
%%% PLot t-SNE of different features for each dataset
%%% 2016-05-11, Yuetan Lin
% close all; clear; clc;

fea_fold = 'X:\wdh\DeepAL_V201508\dataset';

fea_data = {'AwA', 'CUB', 'Dogs'};
fea_mark = {'o', '+'};

fea_indx = [ 1,          2,         3,        4,          5,         6,           7,     8];
fea_name = {'goog1024', 'vgg1000', 'res_fc', 'att_cont', 'wv_g300', 'wv_wcb300', 'eye', 'eye_rand'};
fea_dime = [ 1024,       1000,      1000,     85,         300,       300,         50,    50];
fea_mapp = cell(length(fea_name), 1);
fea_choi = [1 4; 1 5; 1 6; 1 2; 1 3; 1 7; 1 8]; % Feature and Knowledge

seen_rate = [0.5, 0.5, 0.5];
lambda = 1e-4;

errs = zeros(length(fea_data), size(fea_choi,1), 2);
for d = 1:length(fea_data)
    dataset = fea_data{d};
    
    for c = 1:size(fea_choi,1)
        choice = fea_choi(c,:);
        comp_info = sprintf('XK_%s (%s) v.s. (%s)', dataset, fea_name{choice(1)}, fea_name{choice(2)});
        fprintf('===== %d.%d ===== %s ...\n', d, c, comp_info);
        
        %% load data
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(1)} '.mat']);
        try
            load(fea_file, 'VW');
        catch err % Dogs do not have vgg1000 and att_cont
            fprintf('----- Load file not found: %s (%s)\n', dataset, fea_name{choice(1)});
            continue;
        end
        X = VW;
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(2)} '.mat']);
        try
            load(fea_file, 'VW');
        catch err % Dogs do not have vgg1000 and att_cont
            fprintf('----- Load file not found: %s (%s)\n', dataset, fea_name{choice(2)});
            continue;
        end
        K = VW;
        numcls = size(VW,1);
        numseen = floor(numcls * seen_rate(d));
        numunseen = numcls - numseen;
        fprintf('===== Seen: %d, Unseen: %d\n', numseen, numunseen);
        X_S = X(1:numseen,:);
        X_U = X(numseen+1:end,:);
        K_S = K(1:numseen,:);
        K_U = K(numseen+1:end,:);
        
        %% XK_DR
        %W = (X'*X)^(-1)*X'*Y;
        %W = (X'*X+lambda*eye(size(X,2)))^(-1)*X'*K;
        W_S = (X_S'*X_S+lambda*eye(size(X_S,2)))^(-1)*X_S'*K_S;
        X_S = X_S*W_S;
        X_U = X_U*W_S;
        X = [X_S; X_U];
        err_seen = sum(diag(pdist2(K_S, X_S, 'cosine')))/numseen;%norm(K_S-X_S,'fro')/numseen;
        err_unseen = sum(diag(pdist2(K_U, X_U, 'cosine')))/numunseen;
        fprintf('===== SeenErr: %.20f, UnseenErr: %.20f\n', err_seen, err_unseen);
        errs(d, c, :) = [err_seen err_unseen];
        pause(1);
        
        %% t-SNE and show
        % t-SNE - 2. together
        tmp_fea_mapp = tsne([X; K], [1:numcls 1:numcls]);
        fea_mapp{choice(1)} = tmp_fea_mapp(1:numcls,:);
        fea_mapp{choice(2)} = tmp_fea_mapp(numcls+1:end,:);
        
        % show
        hFig = figure(1);clf;
        set(hFig,'units','normalized','outerposition',[0 0 1 1]);
        idx = 1:numcls;
        idx_seen = 1:numseen;
        idx_unseen = numseen+1:numcls;
        subplot(121);
        hold on;
        gscatter([fea_mapp{choice(1)}(idx_seen,1); fea_mapp{choice(2)}(idx_seen,1)], ...
                 [fea_mapp{choice(1)}(idx_seen,2); fea_mapp{choice(2)}(idx_seen,2)], ...
                 [ones(size(idx_seen)) 2*ones(size(idx_seen))], 'rb', 'o+');
        for k = 1:numseen
            x1 = fea_mapp{choice(1)}(idx(k),1);
            y1 = fea_mapp{choice(1)}(idx(k),2);
            x2 = fea_mapp{choice(2)}(idx(k),1);
            y2 = fea_mapp{choice(2)}(idx(k),2);
            line([x1 x2], [y1 y2], 'Color', 'k');
        end
        xlabel(['\fontsize{14}Feature: ' strrep(fea_name{choice(1)}, '_', '\_') ' (' fea_mark{1} ')']);
        ylabel(['\fontsize{14}Knowledge: ' strrep(fea_name{choice(2)}, '_', '\_') ' (' fea_mark{2} ')']);
        title(['\fontsize{16}' strrep(sprintf('SeenErr %.3f', err_seen), '_', '\_')]);
        drawnow;
        hold off;
        subplot(122);
        hold on;
        gscatter([fea_mapp{choice(1)}(idx_unseen,1); fea_mapp{choice(2)}(idx_unseen,1)], ...
                 [fea_mapp{choice(1)}(idx_unseen,2); fea_mapp{choice(2)}(idx_unseen,2)], ...
                 [ones(size(idx_unseen)) 2*ones(size(idx_unseen))], 'rb', 'o+');
        for k = numseen+1:numcls
            x1 = fea_mapp{choice(1)}(idx(k),1);
            y1 = fea_mapp{choice(1)}(idx(k),2);
            x2 = fea_mapp{choice(2)}(idx(k),1);
            y2 = fea_mapp{choice(2)}(idx(k),2);
            line([x1 x2], [y1 y2], 'Color', 'k');
        end
        xlabel(['\fontsize{14}Feature: ' strrep(fea_name{choice(1)}, '_', '\_') ' (' fea_mark{1} ')']);
        ylabel(['\fontsize{14}Knowledge: ' strrep(fea_name{choice(2)}, '_', '\_') ' (' fea_mark{2} ')']);
        title(['\fontsize{16}' strrep(sprintf('UnseenErr %.3f', err_unseen), '_', '\_')]);
        drawnow;
        hold off;
        print('-dpng', sprintf('feature_tSNE_result/%s/%s.png', dataset, comp_info));
        clf;
        %pause();
    end
end
save('feature_tSNE_result/XK_errs.mat', 'errs','fea_data','fea_choi','fea_name');
for d = 1:length(fea_data),fprintf('%s\t',fea_data{d});end;fprintf('\n');
for c = 1:size(fea_choi,1),fprintf('X:%s/K:%s\n',fea_name{fea_choi(c,1)},fea_name{fea_choi(c,2)});end
for d = 1:length(fea_data),for s = 1:2,for c = 1:size(fea_choi,1),fprintf('%.20f\n', errs(d,c,s));end;fprintf('\n');end;fprintf('\n');end