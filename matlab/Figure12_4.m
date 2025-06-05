% Author: Kenji Kashima
% Date  : 2023/11/27

clear;close all; rng(3); % random seed

N_x = 100;  %the number of x in the x_list for depict the orginal function
%the number of samples change it to s_bar=10 to obtain Figure12.3(a)
s_bar = 50; %the number of samples change it to s_bar=50 to obtain Figure12.3(b)

x_bar = linspace(0,1,N_x)'; 
x_sample= rand(s_bar,1) .* rand(s_bar,1); % take x samples randomly

%kernel function                      
ker=@(x,y) min(x*ones(size(y')),ones(size(x))*y'); % eq 13.30

% original function in example 13.1.7 and example 13.1.9
mu_x=@(x) x;  
fn_x=@(x) sin(4*pi*x);

mu_list = mu_x(x_bar);
fn_list = fn_x(x_bar);

% generate samples
sigma = 0.1;
e_s = sigma * randn(s_bar,1);
y_sample=fn_x(x_sample) + e_s; % eq 13.22

K= ker(x_sample,x_sample); % eq 13.24

figure('Name','Figure12.3(a)'); hold on; grid on; 
plot(x_bar,mu_list,'r');  % y=mu(x)
plot(x_bar,fn_list,'k--');  % y=f(x)

%plot samples
scatter(x_sample,y_sample,'b')


kx = ker(x_sample,x_bar);
y_mean = mu_x(x_sample);
mean=mu_x(x_bar)+kx'*(K + sigma^2 * eye(s_bar))^-1*(y_sample-y_mean); % eq 13.16

kxx = ker(x_bar,x_bar);
v=kxx-kx'*(K + sigma^2 * eye(s_bar))^-1*kx; % eq 13.17

vm=sqrt(diag(v));
x_fill = [x_bar', fliplr(x_bar')];        
y_fill = [mean - vm; flipud(mean + vm)]; 

fill(x_fill, y_fill, 'b', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'none'); 
plot(x_bar,mean,'b')

xlabel('$x$','Interpreter','latex','Fontsize',18);
ylabel('$y$','Interpreter','latex','Fontsize',18);
legend('$\mu(x)$','$f(x)$','samples',"","$\mu(x|\mathcal D)$",'Interpreter','latex','Fontsize',16)









