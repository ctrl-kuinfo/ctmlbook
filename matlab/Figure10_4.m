% Author: Kenji Kashima
% Date  : 2025/02/01
% Note  : Optimization Toolbox is required

clear;close all; rng(3); % random seed


% parameter setting
N_k = 1000;
sigma = 1.0;

% transition probability matrix (P0)
P = [1/3,    1/3,      0,      0;
     0  ,    1/3,    1/3,      0;
     0  ,    1/3,    1/3,    1/3;
     2/3,      0,    1/3,    2/3];

[m, n] = size(P);
beta = 0.8;
invbeta = 1/beta;
cost = [1, 2, 3, 4] * sigma;
l = exp(-beta * cost); % l=exp(-cost)

% solve 10.23 func_z
options = optimoptions('fminunc', 'Algorithm','quasi-newton',...
    'OptimalityTolerance',1e-10, 'FunctionTolerance',1e-14,...
    'MaxIterations',1000, 'MaxFunctionEvaluations',5000);
y0 = ones(4,1)*5;
results = fminunc(@(y)func_z(y, beta, cost, P), y0, options);

fprintf("error= %.4f\n", results);
z_opt = exp(results);

% compute P_opt
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


% accumulate P
P_accum = cumsum(P_opt, 1);

% simulation
p_list = zeros(N_k+1, 1, 'int8');
p_list(1) = 4; % initial state
a = [0; 0; 0; 1];
b = [0; 0; 0; 0];
inv_l_hist = zeros(4, N_k);

% constrains (you can change lb, to move the baseline)
options_v = optimoptions('fmincon','Display','off');
lb = [12; 12; 12; 12]; %constrains

for i = 1:N_k
    % maximize the likelyhood func_v 
    v0 = [1; 2; 3; 4]; % can be anything
    results_v = fmincon(@(v)func_v(v, beta, a, b, P), v0,...
        [],[],[],[],lb,[],[],options_v);
    
    inv_v_opt = results_v;
    inv_z_opt = exp(-beta * inv_v_opt);
    inv_l = -log(inv_z_opt.^invbeta ./ (P' * inv_z_opt));
    inv_l_hist(:,i) = inv_l;
    
    % state transition
    u = rand();
    T = P_accum(:, p_list(i)); 
    for j = 1:m
        if u <= T(j)
            p_list(i+1) = j;
            break;
        end
    end
    
    % update count
    a(p_list(i+1)) = a(p_list(i+1)) + 1;
    b(p_list(i)) = b(p_list(i)) + 1;
end

% plot Figure 10.4a
figure;
stairs(0:N_k, double(p_list), 'LineWidth',2);
ylim([0.8, 4.2]);
xlim([0, 50]);
xticks([0, 50]);
xlabel('k');
yticks(1:4);
grid on;

% plot Figure 10.4b
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


% functions
function f = func_z(y, beta, cost, P)
    term = log(P' * exp(y));
    f = sum((y - beta*(-cost(:) + term)).^2);
end

function f = func_v(v, beta, a, b, P)
    term = log(P' * exp(-beta * v));
    f = beta * a' * v + b' * term;
end