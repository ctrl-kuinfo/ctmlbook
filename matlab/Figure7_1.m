% Author: Kenji Kashima
% Date  : 2023/03/12

clear;close all; rng(1); % random seed

% phi is the base function
phi = @(x) [1 ; x ; x^2 ; x^3 ; x^4 ; x^5 ; x^6 ; x^7 ; x^8 ; x^9 ];
% phireal is the function we want to learn
phireal = @(x) 2*sin(5*x); 


n_x = 100;
n_sample = 20;
x = linspace(0,1,n_x);
X = zeros(10,n_x); 
for i=1:n_x
    X(:,i) = phi(x(i)); 
end

figure('Name','Figure 7.1(a)'); hold on; grid on;
for i = 1:n_sample
    theta = randn(10,1); 
    fx = theta' * X;      
    plot(x,fx);          
end
plot(x, zeros(1,10) * X , 'Linewidth' , 3)
xlabel('x','Fontsize',16,'Interpreter','latex')
ylabel('$f({\rm x})$','Fontsize',16,'Interpreter','latex')
set(gca, 'FontName','Times','FontSize',18 ); 

sigma = [0.3,100,10^(-6)]; % hyper parameters
n_data = 8;                % the amount of data
[mu,Sigma,x_s,y_s] = learn_theta(phi,phireal,sigma,n_data);

figure('Name','Figure 7.1(b)'); hold on; grid on;
for i = 1:n_sample
    theta = mvnrnd(mu(:,1),Sigma(:,1:10));
    fx = theta * X;
    plot(x,fx);
end
h(1)=plot( x, mu(:,1)' * X , 'Linewidth' , 3);
h(2)=plot( x, mu(:,2)' * X , 'Linewidth' , 3);
h(3)=plot( x, mu(:,3)' * X , 'Linewidth' , 3);
legend(h([1,2,3]),{'$\sigma=0.3$','$\sigma=10^2$','$\sigma=10^{-6}$'},'Interpreter','latex')

scatter(x_s,y_s,'DisplayName','${\rm y}_s$')
xlabel('x','Fontsize',16,'Interpreter','latex')
ylabel('$f({\rm x})$','Fontsize',16,'Interpreter','latex')
set(gca, 'FontName','Times','FontSize',18 ); 

% learning from n_data in one sample trajectory with hyperparameter sigma
function [mean,cov,x_s,y_s] = learn_theta(phi,phireal,sigma,n_data)
x_s = linspace(0,1,n_data);
X = zeros(10,n_data); y_s=zeros(1,n_data);
for i=1:n_data
    X(:,i) = phi(x_s(i));
    y_s(i) = phireal(x_s(i))+randn ;
end

[~,size_sigma] = size(sigma);
mean = zeros(10,size_sigma);
cov = zeros(10,size_sigma*10);

for i = 1:size_sigma
    tmp = eye(10) - X*minv(X'*X+sigma(i)*eye(n_data))*X';
    cov(:,10*(i-1)+1:10*i) = (tmp+tmp')/2;
    mean(:,i)  = X*minv(X'*X + sigma(i)  *eye(n_data)) *y_s';
end

end

