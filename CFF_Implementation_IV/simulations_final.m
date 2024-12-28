%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Monte Carlo simulations for:                    %%%%%%%%%%
%%%%%%%%% Centorrino, Fève, Florens                       %%%%%%%%%%
%%%%%%%%% ADDITIVE NONPARAMETRIC INSTRUMENTAL REGRESSIONS %%%%%%%%%%
%%%%%%%%% A GUIDE TO IMPLEMENTATION				    	  %%%%%%%%%%
%%%%%%%%% APR 2015                                        %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Authors:                                        %%%%%%%%%%         
%%%%%%%%% Samuele Centorrino 							  %%%%%%%%%%	
%%%%%%%%% (samuele.centorrino[at]stonybrook.edu) 		  %%%%%%%%%%
%%%%%%%%% Frederique Feve   							  %%%%%%%%%%     
%%%%%%%%% (frederique.feve[at]tse-fr.eu)     			  %%%%%%%%%%	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% THE MODEL
% y = phi(z)+ u                       = phi(z) + u
% z = 1/(1 + exp(-(a*w + rho*u + v))) = m(w,v) 

% y et z endogenous (Y on the lhs and Z on the rhs), w instrumental
% variable

zfun = input('Choose the form of the regression function: \n 1) 1-0.3*z \n 2) z.^2 \n 3) sin(3*pi*z) \n');
savestring = ['IM_simulation_data_',num2str(zfun)];

rnds = RandStream('mt19937ar','Seed',5789);

RandStream.setGlobalStream(rnds);

N               = 1000;    

a               = 0.4;

B               = 0.5;

%% Set the number of simulated samples
simit           = input('Enter the number of replications:\n');
K               = 100;

wsave           = zeros(N,simit);
zsave           = zeros(N,simit);
ysave           = zeros(N,simit);

if zfun == 1
	regfun      = @(u) 1-1.5*u;
elseif zfun == 2
	regfun      = @(u) sqrt(2)*u.^2;	
else
	regfun      = @(u) 0.5*sin(1.5*pi*u);
end	

% Optimization options
options         = optimset('TolFun',1.e-08,'TolX',1.e-15,'MaxFunEvals',10000,'MaxIter',500);

%% DEFINE COLORS FOR FINAL PLOT
grey            = [0.2,0.2,0.2];
midgrey         = [0.5,0.5,0.5];
lightgrey       = [0.7,0.7,0.7];

% Preallocate matrices for simulation results
beta_tsls       = zeros(simit,2);
phi_hat_LT_SIM  = zeros(K,simit);
phi_hat_G_SIM   = zeros(K,simit);
phi_hat_LLF_SIM = zeros(K,simit);

alplt_sim       = zeros(simit,1);
bwlt_sim        = zeros(simit,2);
nkg_sim         = zeros(simit,1);
nllf_sim        = zeros(simit,1);
bwllf_sim       = cell(simit,1);

cpu_LT      = zeros(simit,1);
cpu_G       = zeros(simit,1);
cpu_LLF     = zeros(simit,1);

rho         = 4;

for m = 1:simit 
    %%%%%%%%%%%% Draws of w, z and y
    u          = 0.25*randn(N,1);
    v          = 0.5*randn(N,1);

    wsave(:,m) = 5*randn(N,1);
    zeta       = a*wsave(:,m)+rho*u+v;
    
    zsave(:,m) = 1./(1+exp(-zeta)); 

    ysave(:,m) = regfun(zsave(:,m)) + u;
end

%%%%%%%%%%%%%%% The true phi 
zeval    = linspace(max(quantile(zsave,0.01)),min(quantile(zsave,0.99)),K)';
phiv     = regfun(zeval);

save(savestring)

%%%%%%%%%%%%%% Plot the data (NOT RUN)
% &
% xlabel('W')
% ylabel('X')
% print -dpng ./figures/w_z_plot.png
% 
% plot(zsave(:,500),ysave(:,500),'*','Color',lightgrey)
% xlabel('W')
% ylabel('X')
% 
% figstring = ['./figures/z_y_plot',num2str(zfun),'.eps'];
% print('-depsc',figstring)

%%%% Start Estimation
p      = 1;
cste   = 0.5;
itermax= 1000;

parpool
parfor m   = 1:simit
    tirage = [zsave(:,m) ysave(:,m) wsave(:,m)];
    triz   = sortrows(tirage);
    z      = triz(:,1);
    y      = triz(:,2);
    w      = triz(:,3);
    
    %% TSLS estimation
    matz   = [ones(N,1),z];
    matw   = [ones(N,1),w];
    PW     = matw*((matw'*matw)\matw');
    beta_tsls(m,:)= (matz'*PW*matz)\(matz'*PW*y);
    
    %% ESTIMATION WITH TIKHONOV REGULARIZATION
    tic
    [~,alplt_sim(m),phi_hat_LT_SIM(:,m),bwlt] = tikreg(w,z,y,'method','lp','nmulti',2,'zeval',zeval);
	cpu_LT(m)  =toc;
    
    bwlt_sim(m,:)       = [bwlt.hw,bwlt.hz];
end
delete(gcp)
save(savestring)

parpool
parfor m   = 1:simit    
    tirage = [zsave(:,m) ysave(:,m) wsave(:,m)];
    triz   = sortrows(tirage);
    z      = triz(:,1);
    y      = triz(:,2);
    w      = triz(:,3);
    
    %% ESTIMATION WITH GALERKIN REGULARIZATION

    %***** ESTIMATE FOURIER COEFFICIENTS ***** %
    %Optimal number of bases using Horowitz's method
    wnorm     = (w - min(w))/range(w);
    znorm     = (z - min(z))/range(z);
    tic
    nk        = 0;
    crit      = -100;
    while crit < 0
        nk    = nk + 1;
        zz    = legendrep(znorm,nk);
        ww    = legendrep(wnorm,nk+1);
        pmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
        rho2  = 1/real(min(eig(pmat)));
        crit  = (rho2*(nk)^(3.5))/N - 1;
    end
    
    % Pure Galerkin Approach
    qmat      = zz'*(ww*pinv(ww'*ww)*ww')*zz;
    nmat      = zz'*(ww*pinv(ww'*ww)*ww');
    bhat_G    = qmat\(nmat*y);

    %First step estimator of phi
    phi_hat_G      = zz*bhat_G;
    
    Jhatfun = [];
    for j = 1:nk
        zz    = legendrep(znorm,j);
        ww    = legendrep(wnorm,j+1);
        Amin1 = ww*((ww'*(zz*pinv(zz'*zz)*zz')*ww)\ww');
        qmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
        nmat  = zz'*(ww*pinv(ww'*ww)*ww');
        Jhatfun(j) = (2/3)*(log(N)*N^(-2))*sum(((y - phi_hat_G).^2).*sum((Amin1*zz).^2,2)) - sum((zz*(qmat\(nmat*y))).^2);
    end
    
    nk    = find(Jhatfun == min(Jhatfun));
    ww    = legendrep(wnorm,nk+1);
    zz    = legendrep(znorm,nk);
    qmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
    nmat  = zz'*(ww*pinv(ww'*ww)*ww');    

    phi_hat_G_SIM(:,m) = legendrep((zeval-min(zeval))/range(zeval),nk)*(qmat\(nmat*y));

    nkg_sim(m)         = nk;
    
    cpu_G(m)           = toc;
end
delete(gcp)
save(savestring)

nllf = 30;
parpool
parfor m   = 1:simit    
    tirage = [zsave(:,m) ysave(:,m) wsave(:,m)];
    triz   = sortrows(tirage);
    z      = triz(:,1);
    y      = triz(:,2);
    w      = triz(:,3);
    
    %% ESTIMATION WITH LANDWEBER-FRIDMAN REGULARIZATION  
    %[~,nllf,~,bwllf] = lfreg(w,z,y,'method','lp','nmulti',2,'zeval',zeval);
    tic
    
    [Eyw,hw,KWb] = kerreg(w,y,'method','lp','nmulti',2);
    [~,hz,KZb]   = kerreg(z,Eyw,'method','lp','nmulti',2);
    
    [~,~,KZEb]  = kerreg(z,Eyw,'method','lp','xeval',zeval,'par',hz);    
    
    phi_hat_LLF = zeros(K,simit);
    phi_hat_LLF(:,1) = cste*KZEb*Eyw;
    
    stoprule = -1;
    normstop = [];
    bwerror  = hw;
    bzerror  = hz;
    tol      = 1e-05;
    
    phihat0  = cste*KZb*Eyw;
    
    cell(simit,1);
    iter = 0;
    while stoprule < 0 && abs(stoprule) > tol && iter < itermax
        iter      = iter + 1;
        
        [Ephiw,bwerror(iter+1,:),KW1b] = kerreg(w,phihat0,'method','lp','nmulti',2);
           
        [~,bzerror(iter+1,:),KZ1b]= kerreg(z,Eyw - Ephiw,'method','lp','nmulti',2);
		[~,~,KZEb]= kerreg(z,Eyw - Ephiw,'xeval',zeval,'method','lp','par',bzerror(iter+1,:));
		
        phi_hat_LLF(:,iter+1)= phi_hat_LLF(:,iter) + cste*KZEb*(KWb*y - Ephiw);
        
        phihat0 = phihat0 + cste*KZ1b*(Eyw - Ephiw);
        normstop(iter,:) = iter*((Eyw - KWb*phihat0))'*((Eyw - KWb*phihat0));
        
        if iter < nllf
           stoprule = -1;
        else
           stoprule = normstop(iter) - normstop(iter-1);
        end
        if rem(iter,10) == 0
          disp(iter)
        end
    end
    cpu_LLF(m)  =toc;
    
    bwllf_sim{m}        = [bwerror,bzerror];
    [~,vmax]            = max(normstop(1:min(nllf,itermax)));
    [~,vmin]            = min(normstop(vmax:end));
    nllf_sim(m)         = vmin + vmax -1;
    phi_hat_LLF_SIM(:,m)= phi_hat_LLF(:,nllf_sim(m));
    
    disp(m) 
end
delete(gcp)
save(savestring)