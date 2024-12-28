clear
clc

rng(1234);

n = 1000;
rho = 0.01;
gamma = 0.01;
n_g = size(gamma,1);
theta = 5;

rep = 1000;
tsls = NaN(rep,n_g);
ddl = NaN(rep,n_g);
for gg = 1:n_g
    for r = 1:rep
        [tsls(r,gg),ddl(r,gg)] = ddl_sim(n,rho,gamma(gg,1),theta);
    end
end

median(ddl,1)
median(tsls,1)
mean(ddl,1)
mean(tsls,1)
std(ddl,1)
std(tsls,1)