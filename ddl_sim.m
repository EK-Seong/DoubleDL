function [tsls_theta,ddl] = ddl_sim(n,rho,gamma,theta)
    
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
    
    Z = [ones(n,1) t z];
    D = [ones(n,1) t p];
    tsls = (D'*Z/(Z'*Z)*Z'*D)\(D'*Z/(Z'*Z)*Z'*y);
    tsls_theta = tsls(3,1);
    
    % data = [y t t2 t3 t4 z z2 z3  z4 p];
    data = [y p t z];
    % rep = 10;
    % dml = NaN(rep,1);
    K = 2;
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
        disp('second stage and first stage DL prediction rmses')
        disp([rmse2,rmse1])
    
        pfit = firststage.predictFcn(main_sample(:,3:end));
        yfit = secondstage.predictFcn(main_sample(:,3:end));
        
        % z_main = main_sample(:,4);
        % t_main = main_sample(:,3);
        p_main = main_sample(:,2);
        y_main = main_sample(:,1);
        vhat = p_main-pfit;
        yhat = y_main-yfit;
        denom = denom + vhat'*p_main;
        nom = nom + vhat'*yhat;
    end
    ddl = denom\nom;