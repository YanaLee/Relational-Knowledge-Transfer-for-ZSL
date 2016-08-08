%%% PLot t-SNE of different features for each dataset
%%% 2016-05-08, Yuetan Lin
% close all; clear; clc;

fea_fold = 'X:\wdh\DeepAL_V201508\dataset';

fea_data = {'AwA', 'CUB', 'Dogs'};

fea_name = {'vgg1000', 'res_fc', 'goog1024', 'att_cont', 'g300', 'wcb300'};
fea_dime = [ 1000,      1000,     1024,       85,         300,    300];
fea_mark = {'.', 'o', '*', 'x', '+', 's', 'd', '^', 'v'};
fea_mapp = cell(length(fea_name), 1);

for d = 1:length(fea_data)
    dataset = fea_data{d};
    
    fea_choi = [1,2; 1,3; 3,4; 5,3; 5,6];
    for c = 1:size(fea_choi,1)
        if d==3 && c<4 % Dogs do not have vgg1000 and att_cont
            continue;
        end
        choice = fea_choi(c,:);
        comp_info = sprintf('Comparing %s with %s of %s', fea_name{choice(1)}, fea_name{choice(2)}, dataset);
        fprintf('===== %s ...\n', comp_info);
        for n = choice
            fea_file = fullfile(fea_fold, dataset, 'wordvector_mat', [dataset '_' fea_name{n} '.mat']);
            
            if regexp(fea_name{n}, '^(wcb|g)\d+$')
                load(fea_file, ['clsVec' dataset]);
                switch d
                    case 1
                        VW=clsVecAwA;
                    case 2
                        VW=clsVecCUB;
                    case 3
                        VW=clsVecDogs;
                    otherwise
                        error(['Wrong dataset: ' num2str(d)]);
                end
            else
                load(fea_file, 'VW');
            end
            fea_mapp{n} = tsne(VW, []);
        end
        
        % show
        sam_show = 10;
        figure(1);clf;
        hold on;
        ind = randperm(size(VW,1));
        idx = ind(1:sam_show);
        for n = choice
            gscatter(fea_mapp{n}(idx,1), fea_mapp{n}(idx,2), idx, [], fea_mark{n});
        end
        for k = 1:sam_show
            x1 = fea_mapp{choice(1)}(idx(k),1);
            y1 = fea_mapp{choice(1)}(idx(k),2);
            x2 = fea_mapp{choice(2)}(idx(k),1);
            y2 = fea_mapp{choice(2)}(idx(k),2);
            line([x1 x2], [y1 y2], 'Color', 'k');
        end
        xlabel(['\fontsize{14}' strrep(fea_name{choice(1)}, '_', '\_') ' (' fea_mark{choice(1)} ')']);
        ylabel(['\fontsize{14}' strrep(fea_name{choice(2)}, '_', '\_') ' (' fea_mark{choice(2)} ')']);
        title(['\fontsize{18}' strrep(comp_info, '_', '\_')]);
        drawnow;
        print('-dpng', sprintf('feature_tSNE_result/%s/%s.png', dataset, comp_info));
        hold off;
    end
end