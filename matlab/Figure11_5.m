% Author: Kenji Kashima
% Date  : 2025/05/31

clear;close all; rng(3); % random seed

figure('Name','Figure11.5(a)'); hold on; grid on;

x = -5:0.01:5;
y = (x.^2/2-x.*cos(x*10)/20+sin(x*10)/20)/2;
plot(x,y);
y = (x+x.*sin(x*10)/2)/2;
plot(x,y);
y = x+1/2;
plot(x,y)
y = x.*sin(x*10)/2 -1/2;
plot(x,y)
grid on;
xlabel('$x$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
legend('$L(x)$','$\nabla L(x)$','$\nabla L_1(x)$','$\nabla L_2(x)$','Interpreter','latex','Fontsize',10)

N_k = 2000;
C = [1.0, 1.0, 1.0];
alpha = [0.4, 0.8, 1.2];

x_list = zeros(N_k+1,3);
y_list = zeros(N_k,3);
figure('Name','Figure11.5(b)');
hold on; grid on;
x_list(1,:) = 1.0;

for k_param = 1:3
    for i_step = 1:N_k
        x = x_list(i_step,k_param);
        if rand < 0.5
           y_list(i_step,k_param) = x + 1/2;
        else
           y_list(i_step,k_param) = x*sin(x*10)/2 - 1/2;
        end
        x_list(i_step+1,k_param) = x - C(k_param)/((i_step)^alpha(k_param)) * y_list(i_step,k_param);
    end
    plot(0:N_k,x_list(:,k_param));
end
plot(0,1.0,'ko','MarkerFaceColor','k','DisplayName','Initial Value'); % start point - original comment for initial value marker
xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$p_k$','Interpreter','latex','Fontsize',18);
legend_labels = cell(1,3);
for idx = 1:3
    legend_labels{idx} = ['$\alpha=',num2str(alpha(idx)),'$'];
end
legend([legend_labels, {'Initial Value'}],'Interpreter','latex','Fontsize',10,'Location','best');
ylim([-2 2]);
xlim([0, N_k]);
hold off;