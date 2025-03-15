% Author: Kenji Kashima
% Date  : 2023/11/05

clear;close all; rng(3); % random seed


N_k = 1000;

C = [1.0, 1.0];
alpha = [0.4 0.7];

x_list = zeros(N_k,2);
y_list = zeros(N_k,2);
figure('Name','Figure11.5(b)'); 
hold on; grid on;
x_list(1,:) = 2;
for k = 1:2
    for i = 1:N_k    
        x = x_list(i,k);
        if rand < 0.5
           y_list(i,k) = 2*x + randn*10;   %% solving x^2 =2
        else
           y_list(i,k) = x*sin(x*10) + randn*10;   %% solving x^2 =2
        end
        x_list(i+1,k) = x - C(k)/(i^alpha(k)) * y_list(i,k);
    end
    plot(0:N_k,x_list(:,k));
end
plot(0,2,'*'); % start point

xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
legend('$\alpha=0.4$','$\alpha=0.7$','Initial value','Interpreter','latex','Fontsize',10)






