% Author: Kenji Kashima
% Date  : 2023/02/24

clear; close all; rng(1); % random seed
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');


% Figure 3.1(a)
figure('Name', 'Figure 3.1(a)'); hold on; view(-17,22); grid('on');
n_tmp = 1000;
n_k = 9;         % number of k
n_x = 2 * n_tmp + 1; % number of x
x = linspace(-1, 1, n_x);
P = zeros(n_x, n_x);
for i = 1:n_x
    x_i = x(i);
    tmp_u = x_i + 0.1*(x_i - x_i^3) + 0.5*(1 - abs(x_i));
    tmp_l = x_i + 0.1*(x_i - x_i^3) - 0.5*(1 - abs(x_i));
    index_u = ceil((tmp_u + 1) * n_tmp) + 1;
    index_l = floor((tmp_l + 1) * n_tmp) + 1;    
    for j = index_l:index_u
        P(j, i) = P(j, i) + 1 / (index_u - index_l + 1);
    end
end

states = zeros(n_x, n_k + 2); states(n_tmp + 1, 1) = 1;  % initialization
for i = 1:n_k + 1
    states(:, i + 1) = P * states(:, i);
end

for i = 2:n_k + 1
    k = ones(n_x, 1) * (i - 2);
    phi = states(:, i) * n_x / 2;
    plot3(k, x', phi, 'LineWidth', 2)
end
xlabel('$k$', 'Interpreter', 'latex', 'Fontsize', 18);
ylabel('$x$', 'Interpreter', 'latex', 'Fontsize', 18);
zlabel('$\varphi_{x_k}$', 'Interpreter', 'latex', 'Fontsize', 18);

movegui('northeast')


% Figure 3.1(b)
n_k = 50;      % total steps
n_sample = 10; % number of samples 
figure('Name', 'Figure 3.1(b)'); hold on; grid('on'); 

for i = 1:n_sample
    x = zeros(1, n_k); 
    x(1) = rand - 0.5; % initialization
    for k = 1:n_k-1
        x_i = x(k);
        x(k+1) =  x_i + 0.1*(x_i - x_i^3) + 0.5*(rand - 0.5)*(1 - abs(x_i)); 
    end
    plot(x, 'LineWidth', 2)
end
xlim([1, n_k])
xlabel('$k$', 'Interpreter', 'latex', 'Fontsize', 18);
ylabel('$x_k$', 'Interpreter', 'latex', 'Fontsize', 18);
movegui('northwest')
