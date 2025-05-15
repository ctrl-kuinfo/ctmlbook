% Author: Kenji Kashima
% Date  : 2023/11/05
% Note  : Comparison of stochastic models
clear;close all; rng(1); % random seed

N_k = 5000;     % total steps

% the true system y_k = q_k^T * p^*
% where q_k = [y_k, y_{k-1}, y_{k-2}, u_k, u_{k-1}]
coef = conv(conv([1 -0.5],[1 -0.4]),[1 -0.3]); % coefficient of (z-0.5)(z-0.4)(z-0.3)
a = -coef(2:4); n = length(a);
a_prime = a - [0.2,0,0];
b = [1 2];      m = length(b);
p_star = zeros(n+m,N_k); %p^* the true parameter
for i = 1:N_k
    if rem(i,2000)>1000
        p_star(:,i)=[a_prime,b]';
    else
        p_star(:,i)=[a,b]';
    end  
end 

q0 = [zeros(n,1);ones(m,1)]; % initial states and inputs, note that u0=u1=1
p0 = zeros(m+n,1);           % initial parameters are zeors
Sigma0 = 10^4.*eye(m+n);     % See Algorithm 3


% \bar{u} = 1, v_k~(0,1)
sigma_v = 0.1;
v = randn(N_k,1)*sigma_v;     % random noise N(0,sigma_v^2)
u = randn(N_k,1);             % constant input u_k ~ N(0,1)

figure('Name','Figure 8.3(b)'); hold on; grid on;

beta = 1;       % forgetting factor
[~,~,~,p_est,~] = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,beta);
plot(p_est(1,:)); 

beta = 0.995;       % forgetting factor
[~,~,~,p_est,~] = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,beta);
plot(p_est(1,:)); 

beta = 0.8;       % forgetting factor
[~,~,~,p_est,~] = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,beta);
plot(p_est(1,:));

plot(p_star(1,:));  % True parameters

plot(p_star(1),'Linewidth',2); hold on;
legend('$\beta=1$','$\beta=0.995$','$\beta=0.8$','True','Fontsize',16,'Interpreter','latex')
xlabel('$k$','Fontsize',16,'Interpreter','latex')
ylabel('${\rm a}_1$','Fontsize',16,'Interpreter','latex')
ylim([0.8,1.4])



