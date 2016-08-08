function F2K2C = f2k2c_ESZSL(Opt, Data,  gamma, lamda)
% this script used to classification , predict the feature2knowledge2class by ESZSL
%input£ºOpt parameters£¬Dataset£¬ESZSL parameters
%output£ºthe accuracy in Dataset
kesdim = Opt.KES.dim;
acc_train = zeros(kesdim,1);
acc_validation = zeros(kesdim,1);
acc_test = zeros(kesdim,1);
acc_augment = zeros(kesdim,1);

Xta = Data.AugData;       Xtr = Data.TrainData;       Xtv = Data.ValidationData;        Xts = Data.TestData;
Yta = Data.AugLabel_K;    Ytr = Data.TrainLabel_K;    Ytv = Data.ValidationLabel_K;     Yts = Data.TestLabel_K;
Lta = Data.AugLabel_c;    Ltr = Data.TrainLabel_c;    Ltv = Data.ValidationLabel_c;     Lts = Data.TestLabel_c;

k2cMat = double(Opt.KES.anchors);
cls_tr_id = Data.trainclasses_id;  
cls_ts_id = Data.testclasses_id;   
Ktr = k2cMat(cls_tr_id,:); 
Kts = k2cMat(cls_ts_id,:); 
V = solveESZSL(Xtr, Xts, Xta, Ltr, Lts, Lta, k2cMat, cls_tr_id, cls_ts_id, gamma, lamda, Opt.ESZSL.model);

[acc_trp, acc_pc_trp, Otrp, Otr] = predict_eszsl(Xtr, Ltr, V, Ktr, k2cMat, cls_tr_id);
[acc_tvp, acc_pc_tvp, Otvp, Otv] = predict_eszsl(Xtv, Ltv, V, Ktr, k2cMat, cls_tr_id);
[acc_tsp, acc_pc_tsp, Otsp, Ots] = predict_eszsl(Xts, Lts, V, Kts, k2cMat, cls_ts_id);
[acc_tap, acc_pc_tap, Otap, Ota] = predict_eszsl(Xta, Lta, V, Kts, k2cMat, cls_ts_id);
fmtstr = {'%5.2f%%(cls_train)','%5.2f%%(cls_validation)','%5.2f%%(cls_test)','%5.2f%%(cls_aug)'};
if ~isempty(acc_trp), fprintf(['      f2k2c: ',fmtstr{1}], acc_trp); end
if ~isempty(acc_tvp), fprintf(['/', fmtstr{2}], acc_tvp); end
if ~isempty(acc_tsp), fprintf(['/', fmtstr{3}], acc_tsp); end
if ~isempty(acc_tap), fprintf(['/', fmtstr{4}], acc_tap); end
if ~isempty(acc_trp), fprintf('\n'); end
F2K2C.acc_trp = max([acc_trp, 0]);
F2K2C.acc_tvp = max([acc_tvp, 0]);
F2K2C.acc_tsp = max([acc_tsp, 0]);
F2K2C.acc_tap = max([acc_tap, 0]);
F2K2C.acc_pc_trp = acc_pc_trp;
F2K2C.acc_pc_tvp = acc_pc_tvp;
F2K2C.acc_pc_tsp = acc_pc_tsp;
F2K2C.acc_pc_tap = acc_pc_tap;

end

function V = solveESZSL(Xtr, Xts, Xta, Ltr, Lts, Lta, St, cls_tr_id, cls_ts_id, gamma, lamda, ESZSLMode)
Xtr_all = [Xtr;Xta]';
Ltr_all = [Ltr;Lta];
Ztr_all = -1*ones(length(Ltr_all), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Ltr_all)
    c = Ltr_all(i);
    Ztr_all(i,c) = 1;
end
Ztr_all = Ztr_all(:,[cls_tr_id; cls_ts_id]);

Ztr = -1*ones(length(Ltr), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Ltr)
    c = Ltr(i);
    Ztr(i,c) = 1;
end
Ztr = Ztr(:,cls_tr_id);

Zta = -1*ones(length(Lta), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Lta)
    c = Lta(i);
    Zta(i,c) = 1;
end
Zta = Zta(:,cls_ts_id);

Zts = -1*ones(length(Lts), length(unique([cls_tr_id; cls_ts_id])));
for i = 1:length(Lts)
    c = Lts(i);
    Zts(i,c) = 1;
end
Zts = Zts(:,cls_ts_id);

S = St';
S = (S - min(S(:)))/(max(S(:))-min(S(:)));
S_tr = S(:, cls_tr_id);
S_ts = S(:, cls_ts_id);
S_ta = S(:, cls_ts_id);

switch ESZSLMode
    case 0
        V0 = Xtr_all*Xtr_all'+gamma*eye(size(Xtr_all,1));
        V1 = Xtr_all*Ztr_all*S';
        V2 = V1/(S*S'+lamda*eye(size(S,1)));
        V  = V0\V2;

    case 1 % best performance, with our augment data
        V0 = Xtr_all*Xtr_all'+gamma*eye(size(Xtr_all,1));
        V1 = Xtr'*Ztr*S_tr';
        V2 = V1/(S_tr*S_tr'+lamda*eye(size(S_tr,1))); %V1*inv(...)
        V  = V0\V2; %inv(V0)*V2
        
    case 2 % original paper model
        V0 = Xtr'*Xtr+gamma*eye(size(Xtr',1));
        V1 = Xtr'*Ztr*S_tr';
        V2 = V1/(S_tr*S_tr'+lamda*eye(size(S_tr,1)));
        V  = V0\V2;
        
    case 3 % just for test
        V0 = Xta'*Xta+gamma*eye(size(Xta',1));
        V1 = Xta'*Zta*S_ta';
        V2 = V1/(S_ta*S_ta'+lamda*eye(size(S_ta,1)));
        V  = V0\V2;

    case 4 % just for test
        V0 = Xts'*Xts+gamma*eye(size(Xts',1));
        V1 = Xts'*Zts*S_ts';
        V2 = V1/(S_ts*S_ts'+lamda*eye(size(S_ts,1))); %V1*inv(...)
        V  = V0\V2; %inv(V0)*V2
end
end

function [acc_cls, acc_pc, Otxp, Otx] = predict_eszsl(X, L, V, Sx, S, yidx)
acc_cls = []; acc_pc = []; Otxp = []; Otx = [];
if ~isempty(X)
    Otx = X*V;
    A = Otx;
    A = A./repmat(sqrt(diag(A*A')),1, size(A,2));
    S = S./repmat(sqrt(diag(S*S')),1, size(S,2));
    Sx = S(yidx,:);
    L_p = A*Sx';
    [vdump, l_pred] = max(L_p,[],2);
    l_pred = yidx(l_pred);
    acc_cls = 100*sum(l_pred == L)/length(L);
    acc_pc = accPerClass(L, l_pred);
    Otxp = S(l_pred,:);
end
end

function acc_pc = accPerClass(label, pred)
classes_id = sort(unique(label));
acc_pc = zeros(1, length(classes_id));
if isempty(pred) 
    return;
end
for i = 1 : length(classes_id)
    idx = find(label == classes_id(i));
    acc_pc(i) = 100*sum(pred(idx) == classes_id(i))/length(idx);
end
end