% Author: Kenji Kashima
% Date  : 2023/11/05

clear;close all; rng(3); % random seed

N_k = 100;

C = [0.5, 0.5, 1.0, 1.0];
alpha = [1.0 0.3 1.0 1.5];

x_list = zeros(N_k,4);
y_list = zeros(N_k,4);
figure('Name','Figure11.4(a)'); % Please change N_k = 10000 to obtain Figure11.4(b)
hold on; grid on;
x_list(1,:) = 10;
for k = 1:4
    for i = 1:N_k        
        x = x_list(i,k);
        y_list(i,k) = x - randn;   %% mean estimation
        x_list(i+1,k) = x - C(k)/(i^alpha(k)) * y_list(i,k);
    end
    plot(0:N_k,x_list(:,k));
end
plot(0,10,'*'); % start point

xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
legend('$C=0.5,\ \alpha=1$','$C=0.5,\ \alpha=0.3$','$C=1,\ \alpha=1$','$C=0.5,\ \alpha=1.5$','Initial value','Interpreter','latex','Fontsize',10)









