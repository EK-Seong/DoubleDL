%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ESTIMATION OF THE ENGEL CURVE FOR FOOD        			%%%%%%%%
%%%% DATA FROM PAKISTAN                            			%%%%%%%%
%%%% Reference: Bhalotra, Attfield (1998)          			%%%%%%%%
%%%% INTRAHOUSEHOLD RESOURCE ALLOCATION IN RURAL PAKISTAN:  %%%%%%%%
%%%% A SEMIPARAMETRIC ANALYSIS   							%%%%%%%%
%%%% Journal of Applied Econometrics, Vol 13, pp. 463-480   %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Author: Samuele Centorrino 							%%%%%%%%
%%%% Email:  samuele.centorrino[at]stonybrook.edu        	%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

M    = csvread('ba_data.csv');

rnds = RandStream('mt19937ar','Seed',3767);

RandStream.setGlobalStream(rnds);

trim        = quantile(M(:,1),[0.005,0.995]);

M           = M(M(:,1) > trim(1) & M(:,1)< trim(2),:);

fdsh        = M(:,1);
lsize       = M(:,2);
lpcexp      = M(:,3);
lpcinc      = M(:,5);
lpcincsq    = M(:,6);

% Model share = phi(lpcexp) + lsize + quarter dummies (M(:,15:17)) +
% male birth order (M(:,18)) + female birth order (M(:,19)) + 
% household head literacy (M(:,20)) + spouse literacy (M(:,21)) +
% other controls (M(:,22:110))

% Number of bootstrap samples
bootit      = 999;

z           = lpcexp;
w           = lpcinc;
x           = [lsize,M(:,15:21),M(:,37:39),M(:,47:49),M(:,65:67),M(:,75:77)];

clear M

zeval  = linspace(quantile(z,0.005),quantile(z,0.995),100)';

tirage = [z fdsh mksh w x];
triz   = sortrows(tirage);
z      = triz(:,1);
fdsh   = triz(:,2);
mksh   = triz(:,3);
w      = triz(:,4);
x      = triz(:,5:end);
 
N  = numel(fdsh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ESTIMATION USING TIKHONOV REGULARIZATION 				%%%%%%%%
%%% Local Linear Estimator with cv bandwidth   				%%%%%%%%		
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ssr = [];
ssr(1) = 0;
iter= 1;
tol = 1e-05;
crit = 1;
betahat_LLT = zeros(size(x,2),1);
band_LLT = struct('hw',[],'hz',[]);
alpha_LLT = [];

while abs(crit) > tol 
    iter = iter +1;
    [fhat_LLT,alpha_LLT(iter-1),fhat_LLT_eval,band,resid_LLT] = tikreg(w,z,fdsh - x*betahat_LLT,'method','lp','nmulti',2,'zeval',zeval);
    
    band_LLT.hw(iter-1) = band.hw;
    band_LLT.hz(iter-1) = band.hz;
    
    betahat_LLT  = (x'*x)\(x'*(fdsh - fhat_LLT));
    ssr(iter)= mean((fdsh - fhat_LLT-x*betahat_LLT).^2); 
    crit     = ssr(iter) - ssr(iter-1);
end

Bfhat_LLT = zeros(N,bootit);
Bfhat_LLT_eval = zeros(numel(zeval),bootit);
Bbetahat_LLT = zeros(size(x,2),bootit);
for i = 1:bootit
    bsample  = sample([z fdsh w x],N);

    for j = 1:(length(ssr)-1)
        band = struct('hw',band_LLT.hw(j),'hz',band_LLT.hz(j));
        [Bfhat_LLT(:,i),~,Bfhat_LLT_eval(:,i)] = tikreg(bsample(:,3),bsample(:,1),bsample(:,2) - bsample(:,4:end)*Bbetahat_LLT(:,i),'method','lp','spar',band,'rpar',alpha_LLT(j),'zeval',zeval);

        Bbetahat_LLT(:,i) = (bsample(:,4:end)'*bsample(:,4:end))\(bsample(:,4:end)'*(bsample(:,2) - Bfhat_LLT(:,i)));
    end
    disp(i)
end

tbeta_LLT  = betahat_LLT./std(Bbetahat_LLT,[],2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ESTIMATION USING LANDWEBER-FRIDMAN REGULARIZATION 		%%%%%%%%
%%% Local Linear Estimator with cv bandwidth   				%%%%%%%%		
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ssr = [];
ssr(1) = 0;
cste=0.7;
iter= 1;
itermax = 100;
tol = 1e-05;
crit = 1;
betahat_LLLF = zeros(size(x,2),1);
alpha_LLLF = [];
nllf   = 8;

while abs(crit) > tol 
    iter = iter +1;
    
    [Eyw,hw,KWb] = kerreg(w,fdsh - x*betahat_LLLF,'method','lp','nmulti',3);
    [~,hz,KZb]   = kerreg(z,Eyw,'method','lp','nmulti',3);
    
    [~,~,KZEb]   = kerreg(z,Eyw,'method','lp','xeval',zeval,'par',hz);    
    
    fhat_LLLF      = cste*KZb*Eyw;
    fhat_LLLF_eval = cste*KZEb*Eyw;
    
    stoprule = -1;
    normstop = [];
    bwerror  = hw;
    bzerror  = hz;
    %Here 
    j = 0;
    while stoprule < 0 && abs(stoprule) > tol && j < itermax
        j      = j + 1;
        
        [Ephiw,bwerror(j+1,:),KW1b] = kerreg(w,fhat_LLLF(:,j),'method','lp','nmulti',3);
           
        [~,bzerror(j+1,:),KZ1b]= kerreg(z,Eyw - Ephiw,'method','lp','nmulti',3);
		[~,~,KZEb]= kerreg(z,Eyw - Ephiw,'xeval',zeval,'method','lp','par',bzerror(j+1,:));
		
        fhat_LLLF_eval(:,j+1)= fhat_LLLF_eval(:,j) + cste*KZEb*(Eyw - Ephiw);
        
        fhat_LLLF(:,j+1) = fhat_LLLF(:,j) + cste*KZ1b*(Eyw - Ephiw);
        normstop(j,:) = j*((Eyw - KWb*fhat_LLLF(:,j+1)))'*((Eyw - KWb*fhat_LLLF(:,j+1)));
        if j < nllf
           stoprule = -1;
        else
           stoprule = normstop(j) - normstop(j-1);
        end
    end 
    [~,vmax] = max(normstop(1:nllf));
    [~,vmin] = min(normstop(vmax:end));
    
    fhat_LLLF = fhat_LLLF(:,vmin + vmax -1);
    fhat_LLLF_eval = fhat_LLLF_eval(:,vmin + vmax -1);
    
    eval(['band_LLLF.hw',num2str(iter-1),'= bwerror(1:(vmin + vmax -1));'])
    eval(['band_LLLF.hz',num2str(iter-1),'= bzerror(1:(vmin + vmax -1));'])
    
    alpha_LLLF(iter-1) = length(bwerror(1:(vmin + vmax -1)));
    betahat_LLLF  = (x'*x)\(x'*(fdsh - fhat_LLLF));
    ssr(iter)= mean((fdsh - fhat_LLLF-x*betahat_LLLF).^2); 
    crit     = ssr(iter) - ssr(iter-1);
end

Bfhat_LLLF = zeros(N,bootit);
Bfhat_LLLF_eval = zeros(numel(zeval),bootit);
Bbetahat_LLLF = zeros(size(x,2),bootit);
for i = 1:bootit
    bsample  = sample([z fdsh w x],N);
    band     = [];  
    for j = 1:(length(ssr)-1)
        eval(['band = [band_LLLF.hw',num2str(j),',band_LLLF.hz',num2str(j),'];']);
        
        Eyw       = kerreg(bsample(:,3),bsample(:,2) - bsample(:,4:end)*Bbetahat_LLLF(:,i),'method','lp','par',band(1,1));
        [~,~,KZb] = kerreg(bsample(:,1),Eyw,'method','lp','par',band(1,2));

        [~,~,KZEb]= kerreg(bsample(:,1),Eyw,'method','lp','xeval',zeval,'par',band(1,2));    

        Bfhat_LLLF(:,i)     = cste*KZb*Eyw;
        Bfhat_LLLF_eval(:,i)= cste*KZEb*Eyw;
    
        for l = 1:(size(band,1)-1)
            Ephiw = kerreg(bsample(:,3),Bfhat_LLLF(:,i),'method','lp','par',band(l+1,1));

            [~,~,KZ1b]= kerreg(bsample(:,1),Eyw - Ephiw,'method','lp','par',band(l+1,2));
            [~,~,KZEb]= kerreg(bsample(:,1),Eyw - Ephiw,'xeval',zeval,'method','lp','par',band(l+1,2));

            Bfhat_LLLF_eval(:,i)= Bfhat_LLLF_eval(:,i) + cste*KZEb*(Eyw - Ephiw);

            Bfhat_LLLF(:,i) = Bfhat_LLLF(:,i) + cste*KZ1b*(Eyw - Ephiw);
        end
        
        Bbetahat_LLLF(:,i) = (bsample(:,4:end)'*bsample(:,4:end))\(bsample(:,4:end)'*(bsample(:,2) - Bfhat_LLLF(:,i)));
    end
    disp(i)
end

tbeta_LLLF  = betahat_LLLF./std(Bbetahat_LLLF,[],2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ESTIMATION USING SIEVE REGULARIZATION 		            %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%***** Transform Z and W to Unit Interval ***** %
znorm     = (z-min(z))/range(z);
wnorm     = (w-min(w))/range(w);

% Order of the spline  
nk0       = 0;
crit      = -100;
while crit < 0
    nk0   = nk0 + 1;
    zz    = legendrep(znorm,nk0);
    ww    = legendrep(wnorm,nk0 + 1);
    pmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
    rho2  = 1/real(min(eig(pmat)));
    crit  = (rho2*(nk0)^(3.5))/N - 1;
end
 
qmat0     = zz'*(ww*pinv(ww'*ww)*ww')*zz;
nmat0     = zz'*(ww*pinv(ww'*ww)*ww');
    
ssr = [];
ssr(1) = 0;
iter= 1;
tol = 1e-05;
crit = 1;
betahat_G = zeros(size(x,2),1);
nk_G = [];

while abs(crit) > tol 
    iter = iter +1;
    % Pure Galerkin Approach
    ftilde_G  = legendrep(znorm,nk0)*(qmat0\(nmat0*(fdsh-x*betahat_G)));

    Jhatfun = [];
    for j = 1:nk0
        zz    = legendrep(znorm,j);
        ww    = legendrep(wnorm,j+1);
        Amin1 = ww*((ww'*(zz*pinv(zz'*zz)*zz')*ww)\ww');
        qmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
        nmat  = zz'*(ww*pinv(ww'*ww)*ww');
        Jhatfun(j)=(2/3)*(log(N)*N^(-2))*sum(((fdsh - x*betahat_G - ftilde_G).^2).*sum((Amin1*zz).^2,2)) - sum((zz*(qmat\(nmat*(fdsh-x*betahat_G)))).^2);
    end
    
    nk    = find(Jhatfun == min(Jhatfun));
    ww    = legendrep(wnorm,nk+1);
    zz    = legendrep(znorm,nk);
    qmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
    nmat  = zz'*(ww*pinv(ww'*ww)*ww');    
    gamma_G = (qmat\(nmat*(fdsh-x*betahat_G)));
    fhat_G = zz*gamma_G;
    fhat_G_eval = legendrep((zeval-min(zeval))/range(zeval),nk)*gamma_G;
	
    nk_G(iter-1) = nk;
    betahat_G  = (x'*x)\(x'*(fdsh - fhat_G));
    ssr(iter)= mean((fdsh - fhat_G-x*betahat_G).^2); 
    crit     = ssr(iter) - ssr(iter-1);
end

Bfhat_G = zeros(N,bootit);
Bfhat_G_eval = zeros(numel(zeval),bootit);
Bbetahat_G = zeros(size(x,2),bootit);
for i = 1:bootit
    bsample  = sample([z fdsh w x],N);
    
    znorm     = (bsample(:,1)-min(bsample(:,1)))/range(bsample(:,1));
    wnorm     = (bsample(:,3)-min(bsample(:,3)))/range(bsample(:,3));
    for j = 1:(length(ssr)-1)
        ww    = legendrep(wnorm,nk_G(j)+1);
        zz    = legendrep(znorm,nk_G(j));
        qmat  = zz'*(ww*pinv(ww'*ww)*ww')*zz;
        nmat  = zz'*(ww*pinv(ww'*ww)*ww'); 
        Bfhat_G(:,i) = zz*(qmat\(nmat*(bsample(:,2)-bsample(:,4:end)*Bbetahat_G(:,i))));
        Bfhat_G_eval(:,i) = legendrep((zeval-min(zeval))/range(zeval),nk_G(j))*(qmat\(nmat*(bsample(:,2)-bsample(:,4:end)*Bbetahat_G(:,i))));
        
        Bbetahat_G(:,i) = (bsample(:,4:end)'*bsample(:,4:end))\(bsample(:,4:end)'*(bsample(:,2) - Bfhat_G(:,i)));
    end
    disp(i)
end

tbeta_G  = betahat_G./std(Bbetahat_G,[],2);

%% Estimate the quadratic shape using a contol function approach
%% Redefine variables for kernel estimation 
[~,~,KWXb] = kerreg([w,x],z,'par',1.06*std([w,x],[],1)*N^(-1/(4 + size([w,x],2))),...
    'regclass',[0;0;ones(19,1)],'method','lp');

vhat    = z - KWXb*z;

xmat    = [ones(N,1) z z.^2 x vhat];

beta_CF = (xmat'*xmat)\(xmat'*fdsh);
vmat    = ((xmat'*xmat)\eye(size(xmat,2)))*xmat'*diag((fdsh - xmat*beta_CF).^2)*xmat*((xmat'*xmat)\eye(size(xmat,2)));
sder    = sqrt(diag(vmat));
tstat_CF= abs(beta_CF./sder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST DERIVATIVE OF EACH ESTIMATOR      	            %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TIKHONOV
[Eyw,~,KWb] = kerreg(w,fdsh - x*betahat_LLT,'method','lp','par',band_LLT.hw(end));
[~,~,KZb] = kerreg(z,Eyw,'method','lp','par',band_LLT.hz(end));
[~,~,~,fhatprime_LLT_eval] = kerreg(z,(alpha_LLT(end)*eye(N) + KWb*KZb)\Eyw,'method','lp','par',band_LLT.hz(end),'xeval',zeval);

%% LANDWEBER-FRIDMAN
band = [band_LLLF.hw4 band_LLLF.hz4];
[Eyw,~,KWb] = kerreg(w,fdsh - x*betahat_LLLF,'method','lp','par',band(1,1));
[~,~,KZb] = kerreg(z,Eyw,'method','lp','par',band(1,2));

fhat  = cste*KZb*KWb*(fdsh - x*betahat_LLLF);
[~,~,~,fhatprime_LLLF_eval] = kerreg(z,cste*KWb*(fdsh - x*betahat_LLLF),'method','lp','par',band(1,2),'xeval',zeval);
for l = 1:(size(band,1)-1)
    Ephiw = kerreg(w,fhat,'method','lp','par',band(l+1,1));
    [~,~,KZb] = kerreg(z,Ephiw,'method','lp','par',band(l+1,2));
    fhat  = fhat + cste*KZb*(Eyw - Ephiw);
    [~,~,~,temp] = kerreg(z,cste*(Eyw - Ephiw),'method','lp','par',band(l+1,2),'xeval',zeval);
    fhatprime_LLLF_eval = fhatprime_LLLF_eval + temp;
end

%% GALERKIN
fhatprime_G_eval = [zeros(100,1) ones(100,1)/range(zeval) 3*(zeval-min(zeval))/(range(zeval)^2)]*gamma_G;

clear ans Amin1 band bsample bwerror bzerror c01_* c99_* crit Ephiw Eyw ftilde_G hw hz i iter itermax 
clear KW* KZ* l lightgrey nk nk0 nllf nmat nmat0 normstop pmat qmat qmat0 quant rnds rho2 ssr stoprule tirage tol triz trim
clear vmax vmin wnorm ww znorm zz 

save('Engel_Pakistan')

trim     = find(zeval > quantile(z,0.005) & zeval < quantile(z,0.995));
zeval    = zeval(trim);

c01_LLT  = quantile(Bfhat_LLT_eval(trim,:),0.025,2);
c99_LLT  = quantile(Bfhat_LLT_eval(trim,:),0.975,2);

c01_LLLF = quantile(Bfhat_LLLF_eval(trim,:),0.025,2);
c99_LLLF = quantile(Bfhat_LLLF_eval(trim,:),0.975,2);

c01_G    = quantile(Bfhat_G_eval(trim,:),0.025,2);
c99_G    = quantile(Bfhat_G_eval(trim,:),0.975,2);

clear adsh ans Bbetahat_* Bfhat_* chsh cste fdsh j Jhatfun lpcexp lpcexpsq lpcinc
clear lsize mksh N vhat vmat xmat z w x

save('Engel_Pakistan')
