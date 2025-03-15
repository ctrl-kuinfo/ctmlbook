% Author: Kenji Kashima
% Date  : 2023/02/24

clear;close all; rng(1); % random seed

n_sample = 20; % number of samples
n_k = 20;      % number of k

a = 0.5;       % gain
x_max = 10;    % range of y_lim

x_stable = zeros(n_sample,n_k+1); x_noise=zeros(n_sample,n_k+1);
x_stable(:,1) = randn(n_sample,1); x_noise(:,1) = randn(n_sample,1);

step = randn(n_sample,1);

for t = 1:n_k
    noise = randn(n_sample,1);
    x_stable(:,t+1) =  a*x_stable(:,t) + noise*sqrt(3);
    x_noise(:,t+1) = a*x_noise(:,t) + step ;
end


% Figure 3.3(a)
figure('Name','Figure 3.3(a)');grid on;
plot(0:n_k,x_noise','LineWidth',.5); 

xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northeast')

% Figure 3.3(b)
figure('Name','Figure 3.3(b)');grid on;
plot(0:n_k,x_stable','LineWidth',.5); 

xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$x_k$','Fontsize',16,'Interpreter','latex')
ylim([-x_max,x_max]);
movegui('northwest')
