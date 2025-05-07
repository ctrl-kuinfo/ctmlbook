% Author: Kenji Kashima
% Date  : 2025/04/01
% Note  : requires Control System Toolbox

clear; close all;
rng(1);  % Random seed

% Define system matrices
A   = [0.40, 0.37, 0.09;
       0.52, 0.66, 0.15;
       0.21, 0.66, 0.04];
B_u = [0; 1; 0];
B_v = [1; 0; 0];
C   = [1, 0, 0];

% LQR parameters
Q    = diag([0, 0, 1]);
R    = 1;
Qf   = eye(3);
k_bar = 60;

% Noise properties
Rv = 1.0;   % process noise covariance
Rw = 1.0;   % measurement noise covariance

% Initial covariance and state
Sigma0 = 25 * eye(3);
x0     = mvnrnd(zeros(3,1), Sigma0)';
x_max  = 6;

%% LQR backward Riccati recursion
[K, SigmaList] = lqr_control(A, B_u, Q, R, Qf, k_bar);

%% Simulate LQR and LQG
[x_LQR, ~, ~, ~, ~, ~, ~] = simulate_lq_control(A, B_u, B_v, C, Sigma0, K, k_bar, x0, 'lqr', Rw, Rv);
[x_true, x_hat, u, y, Sigmas, x_check, Sigmac] = ...
    simulate_lq_control(A, B_u, B_v, C, Sigma0, K, k_bar, x0, 'lqg', Rw, Rv);

%% Plot Figure 5.3(a)
figure('Name','Figure 5.3(a)'); hold on; grid on;
plot(0:k_bar, x_LQR(3,:), 'k', 'LineWidth',1);
plot(0:k_bar, x_true(3,:), 'b', 'LineWidth',1);
sd = sqrt(squeeze(Sigmas(3,3,:)))';
fill([0:k_bar, fliplr(0:k_bar)], [x_true(3,:)+sd, fliplr(x_true(3,:)-sd)], 'b', 'FaceAlpha',0.2, 'EdgeColor','none');
xlabel('$k$','Interpreter','latex');
legend('$(x_k)_3$ (LQR)','$(x_k)_3$ (LQG)','Location','best','Interpreter','latex');
ylim([-x_max, x_max]);

%% Plot Figure 5.3(b)
figure('Name','Figure 5.3(b)'); hold on; grid on;
plot(0:k_bar, x_true(1,:), 'b', 'LineWidth',1);
plot(0:k_bar-1, y,           'r', 'LineWidth',1);
plot(0:k_bar, x_hat(1,:), 'b--', 'LineWidth',1);
sd = sqrt(squeeze(Sigmas(1,1,:)))';
fill([0:k_bar, fliplr(0:k_bar)], [x_hat(1,:)+sd, fliplr(x_hat(1,:)-sd)], 'b', 'FaceAlpha',0.2, 'EdgeColor','none');
xlabel('$k$','Interpreter','latex');
legend('True $(x_k)_1$','Measurements $y_k$','Estimate $(\hat x_k)_1$','Location','best','Interpreter','latex');
ylim([-x_max, x_max]);

%% Plot Figure 5.3(c)
figure('Name','Figure 5.3(c)'); hold on; grid on;
plot(0:k_bar, x_true(2,:), 'b', 'LineWidth',1);
plot(0:k_bar, x_hat(2,:), 'b--', 'LineWidth',1);
sd = sqrt(squeeze(Sigmas(2,2,:)))';
fill([0:k_bar, fliplr(0:k_bar)], [x_hat(2,:)+sd, fliplr(x_hat(2,:)-sd)], 'b', 'FaceAlpha',0.2, 'EdgeColor','none');
xlabel('$k$','Interpreter','latex');
legend('True $(x_k)_2$','Estimate $(\hat x_k)_2$','Location','best','Interpreter','latex');
ylim([-x_max, x_max]);

%% Plot Figure 5.3(d)
figure('Name','Figure 5.3(d)'); hold on; grid on;
plot(0:k_bar, x_true(3,:), 'b', 'LineWidth',1);
plot(0:k_bar, x_hat(3,:), 'b--', 'LineWidth',1);
sd = sqrt(squeeze(Sigmas(3,3,:)))';
fill([0:k_bar, fliplr(0:k_bar)], [x_hat(3,:)+sd, fliplr(x_hat(3,:)-sd)], 'b', 'FaceAlpha',0.2, 'EdgeColor','none');
xlabel('$k$','Interpreter','latex');
legend('True $(x_k)_3$','Estimate $(\hat x_k)_3$','Location','best','Interpreter','latex');
ylim([-x_max, x_max]);

%% Functions matching Python version
function [K, Sigma] = lqr_control(A, B_u, Q, R, Qf, k_bar)
    Sigma = cell(k_bar+1,1);
    K     = cell(k_bar,1);
    Sigma{k_bar+1} = Qf;
    for k = k_bar:-1:1
        R_tilde = R + B_u' * Sigma{k+1} * B_u;
        S_tilde = A' * Sigma{k+1} * B_u;
        Q_tilde = Q + A' * Sigma{k+1} * A;
        K{k}     = R_tilde \ (S_tilde');
        Sigma{k} = Q_tilde - S_tilde * (R_tilde \ S_tilde');
    end
end

function [x_true, x_hat, u, y, Sigmas, x_check, Sigmac] = simulate_lq_control(A, B_u, B_v, C, Sigma, K, k_bar, x0, mode, Rw, Rv)
    nx = size(A,1);
    w  = sqrt(Rw) * randn(1,k_bar);
    v  = sqrt(Rv) * randn(1,k_bar);
    x_true  = zeros(nx, k_bar+1);
    x_hat   = zeros(nx, k_bar+1);
    Sigmas  = zeros(nx, nx, k_bar+1);
    x_check = zeros(nx, k_bar);
    Sigmac  = zeros(nx, nx, k_bar);
    u       = zeros(1, k_bar);
    y       = zeros(1, k_bar);
    x_true(:,1)   = x0;
    x_hat(:,1)    = zeros(nx,1);
    Sigmas(:,:,1) = Sigma;
    for k = 1:k_bar
        if strcmp(mode,'lqr')
            u(k) = -K{k} * x_true(:,k);
        else
            if strcmp(mode,'lqg_pred')
                u(k) = -K{k} * x_hat(:,k);
            end
            y(k) = C * x_true(:,k) + w(k);
            M_tilde = C * Sigma * C' + Rw;
            L_check = Sigma * C';
            H_check = L_check / M_tilde;
            innov = y(k) - C * x_hat(:,k);
            x_check(:,k) = x_hat(:,k) + H_check * innov;
            Sigma_check = Sigma - L_check / M_tilde * L_check';
            Sigmac(:,:,k) = Sigma_check;
            if strcmp(mode,'lqg')
                u(k) = -K{k} * x_check(:,k);
            end
            x_hat(:,k+1) = A * x_check(:,k) + B_u * u(k);
            Sigma = A * Sigma_check * A' + Rv * eye(nx);
        end
        x_true(:,k+1) = A * x_true(:,k) + B_u * u(k) + B_v * v(k);
        Sigmas(:,:,k+1) = Sigma;
    end
end
