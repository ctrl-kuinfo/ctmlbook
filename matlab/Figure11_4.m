% Author: Kenji Kashima
% Date  : 2025/05/31

clear;close all; rng(3); % random seed

N_k = 100;

C = [0.6, 0.6, 1.0, 1.0];

alpha = [1.0 0.3 1.0 1.5];
x_list = zeros(N_k+1,4); 
y_list = zeros(N_k,4);

figure('Name','Figure11.4(a)'); % Please change N_k = 5000 for Figure11.4(b)
hold on; grid on;
x_list(1,:) = 1.0; 

for k_param = 1:4 
    for i_step = 1:N_k        
        x = x_list(i_step,k_param);
        y_list(i_step,k_param) = x - randn;   % mean estimation: y_k = p_k - z_k
        x_list(i_step+1,k_param) = x - C(k_param)/(i_step^alpha(k_param)) * y_list(i_step,k_param);
    end
    plot(0:N_k,x_list(:,k_param));
end

plot(0,1.0,'ko','MarkerFaceColor','k','DisplayName','Initial Value'); 
xlabel('$k$','Interpreter','latex','Fontsize',18);
ylabel('$p_k$','Interpreter','latex','Fontsize',18); 
legend_labels = cell(1,4);
for idx = 1:4
    legend_labels{idx} = ['$C=',num2str(C(idx)),',\ \alpha=',num2str(alpha(idx)),'$'];
end

legend([legend_labels, {'Initial Value'}],'Interpreter','latex','Fontsize',10,'Location','best')
ylim([-2 2]); 
hold off;