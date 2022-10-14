function [B_db,B_test,B_dbI,B_testI,B_dbT,B_testT] = solve(phi_x1, phi_x2,Phi_testI,Pht_testT,Phi_dbI,Pht_dbT, param,S,L_tr)
[col,row] = size(phi_x1);
[~,LL] = size(L_tr);
n_anchors = param.n_anchors;
mu1 = 0.5;
mu2 = 0.5;
alpha = param.alpha;
beta  = param.beta;
t = param.t;
theta = param.theta;
gamma = param.gamma;
rho = param.rho;
bits = param.bits;
eta = param.eta;

B1 = sign(-1+(1-(-1))*rand(col,bits));
B2 = B1;
R1 =(phi_x1'*phi_x1+gamma*eye(n_anchors+1))\(phi_x1'*B1);
R2 =(phi_x2'*phi_x2+rho*eye(n_anchors+1))\(phi_x2'*B2);

B = [B1,B2];
Z = [B1,B2];
H = B-Z;

C = randn(2*bits,LL);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold = 0.0001;
lastF = 10000000000;
iter = 1;
%% Iterative algorithm
c=[];
while (iter<20)
    Z   =  sign(-alpha*B*C*L_tr'*L_tr*C'+ eta*B + H);
    H   =  H + eta*(B-Z);
    eta =  0.01*eta;
    
    %R1,R2
    R1 = (phi_x1'*phi_x1+gamma*eye(n_anchors+1))\(phi_x1'*B1);
    R2 = (phi_x2'*phi_x2+rho*eye(n_anchors+1))\(phi_x2'*B2);
    
    %C
    C = (alpha*B'*B+beta*eye(2*bits))\(alpha*B'*S*L_tr+beta*B'*L_tr)/(L_tr'*L_tr);
    
    %B
    tt = [mu1*phi_x1*R1,mu2*phi_x2*R2];
    B = sign(tt + 2*alpha*bits*S*L_tr*C'+ 2*beta*L_tr*C' - alpha*Z*C*L_tr'*L_tr*C' + eta*Z - H);
    %B1 B2
    B1 = B(1:col, 1:bits);
    B2 = B(1:col, bits+1:end);
    
    %mu
    h1 = sum(sum((B1-phi_x1*R1).^2));
    h2 = sum(sum((B2-phi_x2*R2).^2));
    mu1 = ((1/h1).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    mu2 = ((1/h2).^(1/(t-1))) / ((1/h1).^(1/(t-1))+(1/h2).^(1/(t-1)));
    
    %W
    Phi_X = [sqrt(mu1)*phi_x1, sqrt(mu2)*phi_x2];
    W = (Phi_X'* Phi_X+theta*eye(2*n_anchors+2))\(Phi_X'*B);
    
    
    norm1 = mean(mean(((B1-phi_x1*R1)).^2))+ gamma*mean(mean(R1).^2);
    norm2 = mean(mean(((B2-phi_x2*R2)).^2))+ rho*mean(mean(R2).^2);
    norm3 = mean(mean((bits*S-B*C*L_tr').^2));
    norm4 = mean(mean((B'-C*L_tr').^2));
    norm5 = mean(mean((B-Phi_X*W).^2))+theta*mean(mean(W).^2);
    
    currentF = norm1 + norm2 + alpha*norm3 + beta*norm4 + norm5;
    fprintf('currentF at iteration %d: %.2f; obj: %.4f\n', iter, currentF, lastF - currentF);     
     disp( currentF);
        if (lastF-currentF)<threshold
            if iter >4
                break
            end
         end
    iter = iter + 1;
    lastF = currentF;
end

Phi_test = [Phi_testI,Pht_testT];
Phi_db = [Phi_dbI,Pht_dbT];
B_db = (Phi_db * W)>0;
B_test = (Phi_test * W)>0;

B1=B(1:col, 1:bits);
R1 =(phi_x1'*phi_x1+gamma*eye(n_anchors+1))\(phi_x1'*B1);
B_dbI = (Phi_dbI *R1)>0;
B_testI = (Phi_testI *R1)>0;

B2=B(1:col, bits+1:end);
R2 =(phi_x2'*phi_x2+rho*eye(n_anchors+1))\(phi_x2'*B2);
B_dbT = (Pht_dbT *R2)>0;
B_testT = (Pht_testT *R2)>0;
end

