% Author: Kenji Kashima
% Date  : 2023/03/12

clear;close all; rng(1); % random seed

% phi is the base function
phi=@(x) [1 ; x ; x^2 ; x^3 ; x^4 ; x^5 ; x^6 ; x^7 ; x^8 ; x^9 ];

% phireal is the function we want to learn
phireal=@(x) 2*sin(5*x); 

% generate data from U(0,1)
n_data = 30;
x_s = rand(n_data,1);
X = zeros(10,n_data); y_s=zeros(n_data,1);
for i=1:n_data
    X(:,i) = phi(x_s(i));
    y_s(i) = phireal(x_s(i))+randn;
end
X=X';

% hyper-parameter
beta = 0.01;

% optimization 1 Naive
cvx_begin
variable theta(10)
minimize(norm(X*theta-y_s))
cvx_end
theta_naive=theta;

% optimization 2 Lasso
cvx_begin
variable theta(10)
minimize(power(2,norm(X*theta-y_s))/n_data+ beta*power(2,norm(theta)))
cvx_end
theta_lasso=theta;

% optimization 3 Ridge
cvx_begin
variable theta(10)
minimize(power(2,norm(X*theta-y_s))/n_data+beta*norm(theta,1))
cvx_end
theta_ridge=theta;

n_x=100;
x = linspace(0,1,n_x);
NAIVE_dat = zeros(1,n_x); LASSO_dat = zeros(1,n_x); 
RIDGE_dat = zeros(1,n_x); f_real = zeros(1,n_x);
for i=1:n_x
    NAIVE_dat(i) = theta_naive'* phi(x(i));
    LASSO_dat(i) = theta_lasso'* phi(x(i));
    RIDGE_dat(i) = theta_ridge'* phi(x(i));
    f_real(i) =  phireal(x(i));
end

figure('Name','Figure 7.4(a)'); hold on;
plot(abs(theta_lasso),'*','MarkerSize',20)
plot(abs(theta_ridge),'.','MarkerSize',20)
xlabel('$i$','Fontsize',16,'Interpreter','latex')
ylabel('$i$-th coefficient','Fontsize',16,'Interpreter','latex')
movegui('northeast')


figure('Name','Figure 7.4(b)'); hold on;
scatter(x_s,y_s)
plot(x,NAIVE_dat,'k','Linewidth',3);
plot(x,LASSO_dat,'r','Linewidth',3);
plot(x,RIDGE_dat,'b','Linewidth',3);
plot(x,f_real,".",'Linewidth',3);
axis([0 1 -5 5])
xlabel('x','Fontsize',16,'Interpreter','latex')
ylabel('$f({\rm x})$','Fontsize',16,'Interpreter','latex')
movegui('northwest')


