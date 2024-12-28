clear
clc
load weak_iv099.mat
t = [mean(ddl,1) mean(tsls,1) ...
    median(ddl,1) median(tsls,1) ...
    std(ddl,1) std(tsls,1)];

subplot(1,2,1)
boxplot(ddl)
xlabel DDL
subplot(1,2,2)
boxplot(tsls)
xlabel 2SLS