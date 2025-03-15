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

function [err_a, err_b,TrSigma,p_est ,y] ...
    = sysid_module(p_star ,n ,q0 ,u ,v ,p0 ,Sigma0 ,beta)
% args: p_star -- the TRUE system parameters [a1,a2,...,a_n,b1,b2,...,b_m]
%       n      -- the number of a  (m denotes the number of b)
%       q0     -- initial data [y_n,...,y_0,u_m,...,u_0]
%       u      -- u_data
%       v      -- v_data
%       p0     -- initial parameters 
%       Sigma0 -- initial Sigma in Algorithm 3
%       beta   -- forgetting factor

[dim_p, N] = size(p_star); % dim_p = m + n  % N denote the length of the data series
y = zeros(1,N-1);  
q = zeros(dim_p,N);
q(:,1) = q0;

% open loop control to genereate y data
for k = 1:N-1
    y(k) =  p_star(:,k)' * q(:,k)+ v(k);
    q(:,k+1) =[y(k);q(1:n-1,k);u(k+1);q(n+1:dim_p-1,k)];
end

p_est = zeros(dim_p,N);
err_a = zeros(1,N);
err_b = zeros(1,N);
TrSigma = zeros(1,N);

p_est(:,1) = p0; 
Sigma = Sigma0;

% sum of the squares between the true parameters and the estimated parameters
err_a(1) = norm(p_star(1:n,1)-p0(1:n))^2;
err_b(1) = norm(p_star(n+1:dim_p,1)-p0(n+1:dim_p))^2;
TrSigma(1) = trace(Sigma);

for k=1:N-1
    % Algorithm 3:
    H = Sigma*q(:,k) / (beta + q(:,k)'*Sigma*q(:,k)); %line 3
    p_est(:,k+1) = p_est(:,k)+H*(y(k)-p_est(:,k)'*q(:,k)); %line 4
    Sigma = (Sigma - H*q(:,k)'*Sigma)/beta; %line 5
 
    % for figure 8.3 (c)~(f)
    err_a(k+1) = norm(p_star(1:n,k+1)-p_est(1:n,k+1))^2;
    err_b(k+1) = norm(p_star(n+1:dim_p,k+1)-p_est(n+1:dim_p,k+1))^2;
    TrSigma(k+1) = trace(Sigma); % trace of Sigma
end

end

