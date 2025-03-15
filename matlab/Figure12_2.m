% Author: Kenji Kashima
% Date  : 2023/11/06
% Note  : You need to install Statistics and Machine Learning Toolbox

clear;close all; rng(3); % random seed

N_x = 100; %the number of x in the x_list
x_list=linspace(0,1,N_x); 

%kernel function
% change it to c=0.1 to obtain Figure12.2(a)
c=0.1;                          % change it to c=1 to obtain Figure13.2(b)
ker=@(x,y) exp(-(x-y').^2/c^2); % eq 13.7 & 13.8 

%original function in example 13.1.7
mu_x=@(x) x;  
y_list=mu_x(x_list');
K=ker(x_list,x_list);

figure('Name','Figure12.2(a)'); hold on; grid on; 
plot(x_list,y_list); % y=mu(x)
for i=1:5 % 5 samples
    plot(x_list,mvnrnd(y_list,K)); %eq 13.13
end

kxx=sqrt(diag(K));

x_fill = [x_list, fliplr(x_list)];        
y_fill = [y_list - kxx; flipud(y_list + kxx)]; 

fill(x_fill, y_fill, 'b', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'none'); 

xlabel('$x$','Interpreter','latex','Fontsize',18);
ylabel('$y$','Interpreter','latex','Fontsize',18);
legend('$\mu(x)$','Interpreter','latex','Fontsize',16)
