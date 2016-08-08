%%% From Feature X to Knowledge K
%%% PLot t-SNE of different features for each dataset
%%% 2016-05-11, Yuetan Lin
% close all; clear; clc;
function test_tSNE_XKPhi_DR()

fea_fold = 'Z:\wdh\DeepAL_V201508\dataset';

fea_data = {'AwA', 'CUB', 'Dogs'};
fea_mark = {'o', '+'};

fea_indx = [ 1,          2,         3,        4,          5,         6,           7,     8];
fea_name = {'goog1024', 'vgg1000', 'res_fc', 'att_cont', 'wv_g300', 'wv_wcb300', 'eye', 'eye_rand'};
fea_dime = [ 1024,       1000,      1000,     85,         300,       300,         50,    50];
fea_mapp = cell(2, 1);
fea_choi = [1 4; 1 5; 1 6; 1 2; 1 3; 1 7; 1 8]; % Feature and Knowledge

seen_rate = [0.5, 0.5, 0.5];
lambda = 1e-6;

errs = zeros(length(fea_data), size(fea_choi,1), 3);
for d = 1:length(fea_data)
    dataset = fea_data{d};
    
    for c = 1:size(fea_choi,1)
        choice = fea_choi(c,:);
        comp_info = sprintf('PhiOmega_%s (%s) v.s. (%s)', dataset, fea_name{choice(1)}, fea_name{choice(2)});
        fprintf('===== %d.%d ===== %s ...\n', d, c, comp_info);
        
        %% load data
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(1)} '.mat']);
        try
            load(fea_file, 'VW');
        catch err % Dogs do not have vgg1000 and att_cont
            fprintf('----- Load file not found: %s (%s)\n', dataset, fea_name{choice(1)});
            continue;
        end
        VW = NormalizeTo_0_1(VW);        
        X = VW;
        fea_file = fullfile(fea_fold, dataset, 'classcenter_mat', [dataset '_' fea_name{choice(2)} '.mat']);
        try
            load(fea_file, 'VW');
        catch err % Dogs do not have vgg1000 and att_cont
            fprintf('----- Load file not found: %s (%s)\n', dataset, fea_name{choice(2)});
            continue;
        end
        VW = NormalizeTo_0_1(VW);
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
        W_S = (X_S'*X_S+lambda*eye(size(X_S,2)))^(-1)*X_S'*K_S; % min ||K_S-X_S*W_S||, (X_dim * K_dim)
        Phi = ((K_S*K_S'+lambda*eye(size(K_S,1)))^(-1)*K_S*K_U')'; % min ||K_U'-K_S'*Phi'||, (numunseen * numseen)
        Omega = ((X_S*X_S'+lambda*eye(size(X_S,1)))^(-1)*X_S*X_U')'; % min ||X_U'-X_S'*Omega'||, (numunseen * numseen)
        X_S = X_S*W_S; % linear mapping of X_S to K_S
        X_U = X_U*W_S; % linear mapping of X_U to K_U
        % error
        err_seen = sum(diag(pdist2(K_S, X_S, 'cosine')))/numseen;%norm(K_S-X_S,'fro')/numseen;
        err_unseen = sum(diag(pdist2(K_U, X_U, 'cosine')))/numunseen;
        cm = norm(diag(1-pdist2(Phi, Omega, 'cosine')),1)/numunseen;
        fprintf('===== SeenErr: %.17f, UnseenErr: %.17f, cm: %.17f\n', err_seen, err_unseen, cm);
        errs(d, c, :) = [err_seen err_unseen cm];
        %pause(1);
        
        %% t-SNE and show
        %{
        % t-SNE - 2. together
        tmp_fea_mapp = tsne([Phi; Omega], [1:numunseen 1:numunseen],2);
        fea_mapp{1} = tmp_fea_mapp(1:numunseen,:);
        fea_mapp{2} = tmp_fea_mapp(numunseen+1:end,:);
        
        % show
        hFig = figure(1);clf;
        set(hFig,'units','normalized','outerposition',[0 0 1 1]);
        idx_unseen = 1:numunseen;
        hold on;
        gscatter([fea_mapp{1}(:,1); fea_mapp{2}(:,1)], ...
                 [fea_mapp{1}(:,2); fea_mapp{2}(:,2)], ...
                 [ones(1,numunseen) 2*ones(1,numunseen)], 'rb', 'o+');
        for k = 1:numunseen
            x1 = fea_mapp{1}(idx_unseen(k),1);
            y1 = fea_mapp{1}(idx_unseen(k),2);
            x2 = fea_mapp{2}(idx_unseen(k),1);
            y2 = fea_mapp{2}(idx_unseen(k),2);
            line([x1 x2], [y1 y2], 'Color', 'k');
        end
        xlabel(['\fontsize{14}Phi (' fea_mark{1} ')']);
        ylabel(['\fontsize{14}Omega (' fea_mark{2} ')']);
        title(['\fontsize{16}' sprintf('cm %.3f', cm)]);
        drawnow;
        hold off;
        print('-dpng', sprintf('feature_tSNE_result/%s/%s.png', dataset, comp_info));
        clf;
        %pause();
        %}
    end
end
save('feature_tSNE_result/XKPhi_errs.mat', 'errs','fea_data','fea_choi','fea_name');
for d = 1:length(fea_data),fprintf('%s\t',fea_data{d});end;fprintf('\n');
for c = 1:size(fea_choi,1),fprintf('X:%s/K:%s\n',fea_name{fea_choi(c,1)},fea_name{fea_choi(c,2)});end
for d = 1:length(fea_data),for s = 1:3,for c = 1:size(fea_choi,1),fprintf('%.17f\n', errs(d,c,s));end;fprintf('\n');end;fprintf('\n');end




function B = NormalizeTo_0_1(A)
v = max(A(:)) - min(A(:));
if v > 0 
    B = (A - min(A(:)))/v;
else
    B = A;
end