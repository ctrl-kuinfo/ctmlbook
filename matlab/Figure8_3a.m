% Author: Kenji Kashima
% Date  : 2023/09/28
% Note  : Comparison of stochastic models
clear;close all; rng(1); % random seed

N_k = 1000;     % total steps
beta = 1;       % forgetting factor

% the true system y_k = q_k^T * p^*
% where q_k = [y_k, y_{k-1}, y_{k-2}, u_k, u_{k-1}]
coef = conv(conv([1 -0.5],[1 -0.4]),[1 -0.3]); % coefficient of (z-0.5)(z-0.4)(z-0.3)
a = -coef(2:4); n = length(a);
b = [1 2];      m = length(b);
p_star = zeros(n+m,N_k); %p^* the true parameter
for i = 1:N_k
    p_star(:,i)=[a,b]';
end 

q0 = [zeros(n,1);ones(m,1)]; % initial states and inputs, note that u0=u1=1
p0 = zeros(m+n,1);           % initial parameters are zeors
Sigma0 = 10^4.*eye(m+n);     % See Algorithm 3

figure('Name','Figure 8.3(a)'); hold on; grid on;
% u_k = 1, v_k~N(0,1)
sigma_v = 1;
v = randn(N_k,1)*sigma_v;     % random noise N(0,sigma_v^2)
u = ones(N_k,1);              % constant input u_k = 1
[~,~,~,~,y] = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,beta);
plot(y);
% u_k ~ N(1,1), v_k~N(0,1)
sigma_v = 0.1;
v = randn(N_k,1)*sigma_v;     % random noise N(0,sigma_v^2)
u = ones(N_k,1)+randn(N_k,1); % random input N(1,1)

[~,~,~,~,y] = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,beta);
plot(y);
legend('$u_k=1,v_k\sim {\mathcal N}(0,1)$',['$u_k \sim {\mathcal N}(1,1),' ...
    'v_k\sim {\mathcal N}(0,0.01)$'],'Fontsize',16,'Interpreter','latex')
xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('$y_k$','Fontsize',16,'Interpreter','latex')


