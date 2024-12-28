clear
clc

rng(1234);

n = 1000;
rho = 0.5;
gamma = 0.001;
theta = 2;
t = unifrnd(0,10,n,1);
psi = NaN(n,1);
for i=1:n
    psi(i,1) = price_sensitivity(t(i,1));
end
z = mvnrnd(0,1,n);
ue = mvnrnd([0,0],[[1,rho];[rho,1]],n);
u = ue(:,1);
e = ue(:,2);
p = 25*ones(n,1)+gamma*z.*psi+3*psi+u;
y = 100*ones(n,1)+10*psi-theta*p+e;

% t2 = t.^2;
% t3 = t.^3;
% t4 = t.^4;
% z2 = z.^2;
% z3 = z.^3;
% z4 = z.^4;

Z = [ones(n,1) t z];
D = [ones(n,1) t p];
tsls = (D'*Z/(Z'*Z)*Z'*D)\(D'*Z/(Z'*Z)*Z'*y);

% data = [y t t2 t3 t4 z z2 z3  z4 p];
data = [y p t z];
% rep = 10;
% dml = NaN(rep,1);
K = 2;
% for r = 1:rep
    idx = randperm(n)';
    jj = 1:1:n;
    jj = jj';
    denom = 0;
    nom = 0;
    for i=1:K
        main_sample_index = jj >= n/K*(i-1)+1 & jj <= n/K*i;
        aux_sample_index = logical(1-main_sample_index);
        aux_sample = data(idx(aux_sample_index),:);
        main_sample = data(idx(main_sample_index),:);
        
    
        [secondstage,rmse2] = DNN2(aux_sample);
        [firststage,rmse1] = DNN1(aux_sample);
    
        pfit = firststage.predictFcn(main_sample(:,3:end));
        yfit = secondstage.predictFcn(main_sample(:,3:end));
        
        z_main = main_sample(:,4);
        t_main = main_sample(:,3);
        p_main = main_sample(:,2);
        y_main = main_sample(:,1);
        vhat = p_main-pfit;
        yhat = y_main-yfit;
        denom = denom + vhat'*p_main;
        nom = nom + vhat'*yhat;
    end
    dml = denom\nom;
    % dml(r,1) = denom\nom;
% end
% median(dml)
% % test 
% pgrid = 12:0.01:28;
% tt = 1;
% zz = 0;
% 
% prediction = NaN(size(pgrid,2),1);
% counterfactual = NaN(size(pgrid,2),1);
% for ii =1:size(pgrid,2)
%     prediction(ii,1) = beta'*[1; tt; tt^2; tt^3; pgrid(1,ii);pgrid(1,ii)^2;pgrid(1,ii)^3];
%     counterfactual(ii,1) = 100 + (10+pgrid(1,ii))*(2*((tt-5)^4/600+exp(-4*(tt-5)^2)+tt/10-2))-2*pgrid(1,ii);
% end
% 
% scatter(p,y); hold on;
% scatter(pgrid,prediction); hold on;
% scatter(pgrid,counterfactual);hold off;
% legend 'observation' 'NPIV' 'true'
% 
% scatter(p,y); hold on;
% scatter(pgrid,counterfactual); hold off;

% n = 500;
% rho = 0.5;
% t = unifrnd(0,10,n,1);
% psi = 2*((t-5*ones(n,1)).^4/600+exp(-4*(t-5*ones(n,1)).^2)+t/10-2*ones(n,1));
% z = mvnrnd(0,1,n);
% nu = mvnrnd(0,1,n);
% e = NaN(n,1);
% for i=1:n
%     e(i,1) = mvnrnd(rho*nu(i,1),1-rho^2,1);
% end
% p = 25*ones(n,1)+(z+3*ones(n,1).*psi+nu);
% y=100*ones(n,1)+(10*ones(n,1)+p).*psi-2*p+e;
% 
% t2 = t.^2;
% t3 = t.^3;
% z2 = z.^2;
% z3 = z.^3;
% p2 = p.^2;
% p3 = p.^3;
% 
% 
% D = [ones(n,1) t t2 t3 p p2 p3];
% prediction = D*beta;
% target = 100*ones(n,1)+(10*ones(n,1)+p).*psi-2*p;
% rmse = sqrt(((prediction-target)'*(prediction-target))/n);
% 
% w1 = trainedModel.RegressionNeuralNetwork.LayerWeights(1,1);
% w2 = trainedModel.RegressionNeuralNetwork.LayerWeights(1,2);
% w3 = trainedModel.RegressionNeuralNetwork.LayerWeights(1,3);
% w1 = cell2mat(w1);
% w2 = cell2mat(w2);
% w3 = cell2mat(w3);
% b1 = trainedModel.RegressionNeuralNetwork.LayerBiases(1,1);
% b2 = trainedModel.RegressionNeuralNetwork.LayerBiases(1,2);
% b3 = trainedModel.RegressionNeuralNetwork.LayerBiases(1,3);
% b1 = cell2mat(b1);
% b2 = cell2mat(b2);
% b3 = cell2mat(b3);
% 
% Z = [t t2 t3 z z2 z3 ones(n,1)];
% 
% l1 = (Z*w1'+ kron(ones(n,1),b1'));
% a1 = NaN(n,size(w2,2));
% for ii = 1:n
%     for jj = 1:size(w2,2)    
%         a1(ii,jj) = relu(l1(ii,jj));
%     end
% end
% 
% l2 = a1*w2'+ kron(ones(n,1),b2');
% a2 = NaN(n,size(w3,2));
% for ii = 1:n
%     for jj = 1:size(w3,2)    
%         a2(ii,jj) = relu(l2(ii,jj));
%     end
% end
% 
% w1 = trainedModel1.RegressionNeuralNetwork.LayerWeights(1,1);
% w2 = trainedModel1.RegressionNeuralNetwork.LayerWeights(1,2);
% w3 = trainedModel1.RegressionNeuralNetwork.LayerWeights(1,3);
% w1 = cell2mat(w1);
% w2 = cell2mat(w2);
% w3 = cell2mat(w3);
% b1 = trainedModel1.RegressionNeuralNetwork.LayerBiases(1,1);
% b2 = trainedModel1.RegressionNeuralNetwork.LayerBiases(1,2);
% b3 = trainedModel1.RegressionNeuralNetwork.LayerBiases(1,3);
% b1 = cell2mat(b1);
% b2 = cell2mat(b2);
% b3 = cell2mat(b3);
% 
% l11 = (Z*w1'+ kron(ones(n,1),b1'));
% a11 = NaN(n,size(w2,2));
% for ii = 1:n
%     for jj = 1:size(w2,2)    
%         a11(ii,jj) = relu(l11(ii,jj));
%     end
% end
% 
% l21 = a11*w2'+ kron(ones(n,1),b2');
% a21 = NaN(n,size(w3,2));
% for ii = 1:n
%     for jj = 1:size(w3,2)    
%         a21(ii,jj) = relu(l21(ii,jj));
%     end
% end
% 
% beta_deep = ((a21'*a2)-(10*eye(10)))\(a21'*y);