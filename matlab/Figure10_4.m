% Author: Kenji Kashima
% Date  : 2025/05/31
% Note  : Optimization Toolbox is required

clear;close all; rng(3); % random seed

% parameter setting
N_k = 1000;
sigma_param = 1.0; % Renamed from sigma

% transition probability matrix (P0)
P0 = [1/3,    1/3,     0,     0; % Renamed from P
      0  ,    1/3,   1/3,     0;
      0  ,    1/3,   1/3,   1/3;
      2/3,      0,   1/3,   2/3];

[m, n] = size(P0);
beta = 0.8;
invbeta = 1/beta;
cost_val = [1, 2, 3, 4] * sigma_param; % Renamed from cost

% solve 10.23 L_KL
options = optimoptions('fminunc', 'Algorithm','quasi-newton',...
    'OptimalityTolerance',1e-10, 'FunctionTolerance',1e-14,...
    'MaxIterations',1000, 'MaxFunctionEvaluations',5000);
v_init_kl = ones(4,1)*5; % Initialization for for V* optimization in L_KL
V_star_optimal = fminunc(@(v_star)L_KL(v_star, beta, cost_val, P0), v_init_kl, options); % Changed to call L_KL, V_star_optimal is V*

fprintf("error= %.4f\n", V_star_optimal); % This 'error' is the optimized V* vector
z_opt = exp(-beta * V_star_optimal); % Calculate Z* from V*

% compute P_opt
P_opt = zeros(size(P0));
for r_idx = 1:m % Renamed from i
    for c_idx = 1:n % Renamed from j
        P_opt(r_idx,c_idx) = P0(r_idx,c_idx) * z_opt(r_idx) / (P0(:,c_idx)' * z_opt); % Corrected P_opt calculation based on Z*
    end
end

fprintf("sigma: %.1f\nPpi:\n", sigma_param);
disp(P_opt);

% 固有値と固有ベクトルを計算
[V, D] = eig(P_opt);  % V: 固有ベクトル（列ベクトル）, D: 固有値の対角行列
% 固有値1に最も近い固有値を探す（通常、確率行列の場合は1に等しい）
[~, index] = max(real(diag(D))); 
eigenvector_for_1 = real(V(:, index));  % 実数部を抽出
% 要素の和で正規化して定常分布を得る
sum_of_elements = sum(eigenvector_for_1);
p_stationary = eigenvector_for_1 / sum_of_elements;

disp("p_stationary for P_opt：");
disp(p_stationary);

% p_stationary = ones(4, 1) / 4; % Renamed from p_stable
% for i_iter = 1:100 % Renamed from i
%     p_stationary = P_opt * p_stationary;
% end
% fprintf('p_stationary_100 = \n'); % Renamed
% disp(p_stationary);


% accumulate P
P_accum = cumsum(P_opt, 1);

% simulation
p_list = zeros(N_k+1, 1, 'int8');
p_list(1) = 4; % initial state
a_counts = [0; 0; 0; 1]; % Renamed from a
b_counts = [0; 0; 0; 0]; % Renamed from b
inv_l_hist = zeros(4, N_k);

% constrains (you can change lb, to move the baseline)
options_v_fmincon = optimoptions('fmincon','Display','off'); % Renamed from options_v
lb_v = [12; 12; 12; 12]; % Renamed from lb %constrains - original comment kept

for i_main = 1:N_k % Renamed from i
    % maximize the likelyhood L_IRL 
    v0_irl = [1; 2; 3; 4]; % Renamed from v0 % can be anything - original comment kept
    V_star_estimated_irl = fmincon(@(v_star_irl)L_IRL(v_star_irl, beta, a_counts, b_counts, P0), v0_irl,... % Changed to call L_IRL
        [],[],[],[],lb_v,[],[],options_v_fmincon);
    
    inv_v_opt_irl = V_star_estimated_irl; % Renamed from inv_v_opt
    inv_z_opt_irl = exp(-beta * inv_v_opt_irl); % Renamed from inv_z_opt
    inv_l_estimated = -log(inv_z_opt_irl.^invbeta ./ (P0' * inv_z_opt_irl)); % Renamed from inv_l
    inv_l_hist(:,i_main) = inv_l_estimated;
    
    % state transition
    u_rand = rand(); % Renamed from u
    T_accum = P_accum(:, p_list(i_main));  % Renamed from T
    for j_state = 1:m % Renamed from j
        if u_rand <= T_accum(j_state)
            p_list(i_main+1) = j_state;
            break;
        end
    end
    
    % update count
    a_counts(p_list(i_main+1)) = a_counts(p_list(i_main+1)) + 1;
    b_counts(p_list(i_main)) = b_counts(p_list(i_main)) + 1;
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
for i_plot = 1:4 % Renamed from i
    plot(inv_l_hist(i_plot,:)-inv_l_hist(1,:)+1, 'DisplayName', sprintf('$\\ell_{%d}$', i_plot));
end
ylim([0, 6]);
xlim([0, N_k]);
xlabel('k');
legend('Interpreter','latex');
grid on;


% functions
% Loss function for KL control, and it now optimizes V* directly
function sum_sq_residual = L_KL(V_star_var, beta_param, cost_val_param, P0_matrix) 
    term_log = log(P0_matrix' * exp(-beta_param * V_star_var));
    residuals = V_star_var - cost_val_param(:) + term_log;
    sum_sq_residual = sum(residuals.^2);
end

% Loss Function for IRL
function neg_log_likelihood = L_IRL(V_star_var, beta_param, a_counts_param, b_counts_param, P0_matrix) 
    term_log = log(P0_matrix' * exp(-beta_param * V_star_var));
    neg_log_likelihood = beta_param * a_counts_param' * V_star_var + b_counts_param' * term_log;
end