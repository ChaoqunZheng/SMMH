function [] = demo_SMMH(bits, dataname)
%myFun - Description
%
% Syntax: [] = demo_SMMH(bits, dataname)
%
% Long description

warning off;
%% Dataset Loading
addpath('../../Data');

if strcmp(dataname, 'flickr')
    [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr] = construct_mir();
elseif strcmp(dataname, 'nuswide')
     [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr] = construct_nus();
elseif strcmp(dataname, 'coco')
    [I_te, T_te, L_te, I_db, T_db, L_db, I_tr, T_tr, L_tr] = construct_coco();
else
    fprintf('ERROR dataname!');
end

if strcmp(dataname, 'flickr')
    alpha = 1e-3;
    beta = 1e-3;
    theta = 1e-3;
    gamma = 1e-3;
    rho = 1e-3;
    eta = 1e3;
    n_anchors = 500;
elseif strcmp(dataname, 'nuswide')
    alpha = 1e1;
    beta = 1e-3;
    theta = 1e-2;
    gamma = 1e-5;
    rho = 1e-3;
    eta = 1e3;
    n_anchors = 800;
elseif strcmp(dataname, 'coco')
   alpha = 1e-5;
    beta = 1e-1;
    theta = 1e-2;
    gamma = 1e-5;
    rho = 1e-1;
    eta = 1e-1;
    n_anchors =800;
else
    fprintf('ERROR paramethers!');
end

%% Training & Evaluation Process
fprintf('\n============================================Start training CHMR-our============================================\n');
run = 1;
bits = str2num(bits);

param.alpha = alpha;
param.beta = beta;
param.theta = theta;
param.gamma = gamma;
param.rho = rho;
param.n_anchors = n_anchors;
param.t = 2;
param.bits = bits;
param.eta = eta;

mapITIT = zeros(run,1);
mapII = zeros(run,1);
mapTT = zeros(run,1);
mapTI = zeros(run,1);
mapIT = zeros(run,1);

%% Data preparing

Ntrain = size(I_tr,1);
S = 2*L_tr*L_tr'-ones(Ntrain);
sample = randsample(Ntrain, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);

M1=mean(anchorI,1);
M2=mean(anchorT,1);
PP1=anchorI- repmat(M1,n_anchors,1);
PP2=anchorT- repmat(M2,n_anchors,1);
v1=sum(sum(PP1.^2));
v2=sum(sum(PP2.^2));
sigmaI=sqrt(v1/n_anchors-1);
sigmaT=sqrt(v2/n_anchors-1);
%             sigmaI = 67;
%             sigmaT = 2;
PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhtT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhtT = [PhtT, ones(Ntrain,1)];

Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
Pht_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
Pht_testT = [Pht_testT, ones(size(Pht_testT,1),1)];

Phi_dbI = exp(-sqdist(I_db,anchorI)/(2*sigmaI*sigmaI));
Phi_dbI = [Phi_dbI, ones(size(Phi_dbI,1),1)];
Pht_dbT = exp(-sqdist(T_db,anchorT)/(2*sigmaT*sigmaT));
Pht_dbT = [Pht_dbT, ones(size(Pht_dbT,1),1)];
S = bits*S;
S(S==0) = -bits;

for j = 1:run
   
    [B_db,B_test,B_dbI,B_testI,B_dbT,B_testT] = solve(PhiI, PhtT,Phi_testI,Pht_testT,Phi_dbI,Pht_dbT, param,S,L_tr);

    B_db=compactbit(B_db);
    B_test=compactbit(B_test);
    B_dbI=compactbit(B_dbI);
    B_testI=compactbit(B_testI);
    B_dbT=compactbit(B_dbT);
    B_testT=compactbit(B_testT);
   
    fprintf('start evaluating...\n');
    DhITIT = hammingDist(B_db, B_test);
    [P1] = perf_metric4Label( L_db, L_te, DhITIT);
    mapITIT=P1;
    
    
    DhTI = hammingDist(B_dbI, B_testT);
    [P4] = perf_metric4Label( L_db, L_te, DhTI);
    mapTI(j) = P4;
    
    DhIT= hammingDist(B_dbT, B_testI);
    [P5] = perf_metric4Label( L_db, L_te, DhIT);
    mapIT(j) = P5;
    
end

 
fprintf('[%s-%s] ITIT MAP = %.4f\n', dataname, num2str(bits), mean(mapITIT));
fprintf('[%s-%s] TI MAP = %.4f\n', dataname, num2str(bits), mean(mapTI));
fprintf('[%s-%s] IT MAP = %.4f\n', dataname, num2str(bits),  mean(mapIT));
end
