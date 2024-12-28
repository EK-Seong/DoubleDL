clear
clc

%rng(1234);

n = [100 150 200 250 300 450 500];
n = n';
rho = 0.9;
c = 0.0001;
gamma = [c/sqrt(100);c/sqrt(150);c/sqrt(200);c/sqrt(250);c/sqrt(300);c/sqrt(350);c/sqrt(400);c/sqrt(450);c/sqrt(500)];
n_N = size(n,1);
theta = 2;

rep = 1;
tsls = NaN(rep,n_N);
ddl = NaN(rep,n_N);
for gg = 1:n_N
    for r = 1:rep
        [tsls(r,gg),ddl(r,gg)] = ddl_sim(n(gg,1),rho,gamma(gg,1),theta);
    end
end

median(ddl,1)
median(tsls,1)
mean(ddl,1)
mean(tsls,1)