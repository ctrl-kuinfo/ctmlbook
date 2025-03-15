% Author: Kenji Kashima
% Date  : 2025/02/01
% Note  : Optimization Toolbox & CVX are required

clear; close all; rng(3); % random seed

% Parameter setting
N_k = 2000;
sigma = 1.0;

% Transition probability matrix (P0)
P = [1/3, 1/3, 0, 0;
     0, 1/3, 1/3, 0;
     0, 1/3, 1/3, 1/3;
     2/3, 0, 1/3, 2/3];

[m, n] = size(P);
beta = 0.8;
invbeta = 1/beta;
cost = [1, 2, 3, 4] * sigma;
l = exp(-beta * cost);

% Solve 10.23 func_z using CVX
cvx_begin quiet
    variable y(m)
    term = log(P' * exp(y));
    minimize(sum((y - beta*(-cost(:) + term)).^2))
cvx_end

z_opt = exp(y);

% Compute P_opt
P_opt = zeros(size(P));
for i = 1:m
    for j = 1:n
        P_opt(i,j) = P(i,j) / (P(:,j)' * (z_opt / z_opt(i)));
    end
end

fprintf("sigma: %.1f\nPpi:\n", sigma);
disp(P_opt);

p_stable = ones(4, 1) / 4;
for i = 1:100
    p_stable = P_opt * p_stable;
end

% closed loop stable state distribution
fprintf('p_100 = \n');
disp(p_stable);

% Accumulate P
P_accum = cumsum(P_opt, 1);

% Simulation
p_list = zeros(N_k+1, 1, 'int8');
p_list(1) = 4; % Initial state
a = [0; 0; 0; 1];
b = [0; 0; 0; 0];
inv_l_hist = zeros(4, N_k);

% Constraints (you can change lb to move the baseline)
lb = [12; 12; 12; 12];

for i = 1:N_k
    % Maximize the likelihood func_v using CVX
    cvx_begin quiet
        variable v(m)
        term = log(P' * exp(-beta * v));
        minimize(beta * a' * v + b' * term)
        subject to
            v >= lb;
    cvx_end
    
    inv_v_opt = v;
    inv_z_opt = exp(-beta * inv_v_opt);
    inv_l = -log(inv_z_opt.^invbeta ./ (P' * inv_z_opt));
    inv_l_hist(:,i) = inv_l;
    
    % State transition
    u = rand();
    T = P_accum(:, p_list(i)); 
    for j = 1:m
        if u <= T(j)
            p_list(i+1) = j;
            break;
        end
    end
    
    % Update count
    a(p_list(i+1)) = a(p_list(i+1)) + 1;
    b(p_list(i)) = b(p_list(i)) + 1;
end

% Plot Figure 10.4a
figure;
stairs(0:N_k, double(p_list), 'LineWidth',2);
ylim([0.8, 4.2]);
xlim([0, 50]);
xticks([0, 50]);
xlabel('k');
yticks(1:4);
grid on;

% Plot Figure 10.4b
figure;
hold on;
for i = 1:4
    plot(inv_l_hist(i,:)-inv_l_hist(1,:)+1, 'DisplayName', sprintf('$\\ell_{%d}$', i));
end
ylim([0, 6]);
xlim([0, N_k]);
xlabel('k');
legend('Interpreter','latex');
grid on;
