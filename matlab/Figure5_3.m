% Author: Kenji Kashima
% Date  : 2024/10/12
% Note  : Example 5.2.7

clear;close all; rng(1); % random seed


% System matrices
A = [0.40, 0.37, 0.09;
     0.52, 0.66, 0.15;
     0.21, 0.66, 0.04];  
B_u = [0; 1; 0];         % Control input matrix
B_v = [1; 0; 0];         % Disturbance input matrix (noise)
C = [1, 0, 0];           % Output matrix

% LQR parameters
Q = diag([0, 0, 1]);     % State cost matrix
R = 1;                   % Control input cost
Qf = eye(3);             % Final state cost
k_bar = 60;              % Total time steps

% Noise properties
Rv = 1;                  % Process noise covariance (v_k ~ N(0, 1))
Rw = 4;                  % Measurement noise covariance (w_k ~ N(0, 4))

% Initialize state and control matrices for finite-horizon LQR
P = cell(k_bar+1, 1);    % Store cost-to-go matrices P
K = cell(k_bar, 1);      % Store feedback gains K
P{k_bar+1} = Qf;         % Terminal cost is Qf

% Backward recursion to solve the finite-horizon LQR problem
for k = k_bar:-1:1
    K{k} = (R + B_u' * P{k+1} * B_u) \ (B_u' * P{k+1} * A);
    P{k} = Q + A' * P{k+1} * A - A' * P{k+1} * B_u * K{k};
end

% Design Kalman filter (state estimator) for LQG control
% Solve Riccati equation for Kalman gain
S = cell(k_bar+1, 1);    % Store state estimation error covariance
L = cell(k_bar, 1);      % Store Kalman gains
S{k_bar+1} = Qf;         % Terminal state estimation error covariance

for k = k_bar:-1:1
    % Compute Kalman gain
    L{k} = A * S{k+1} * C' / (C * S{k+1} * C' + Rw);
    % Riccati recursion for state estimation error covariance
    S{k} = A * S{k+1} * A' - A * S{k+1} * C' / (C * S{k+1} * C' + Rw) * C * S{k+1} * A' + B_v * Rv * B_v';
end

% Simulation for LQR and LQG controllers
x_LQR = zeros(3, k_bar+1);   % True state for LQR control
x_LQG = zeros(3, k_bar+1);   % True state for LQG control
x_hat_LQG = zeros(3, k_bar+1);  % Estimated state for LQG control
u_LQR = zeros(1, k_bar);     % Control inputs for LQR
u_LQG = zeros(1, k_bar);     % Control inputs for LQG
y_LQG = zeros(1, k_bar);     % Output measurements for LQG
w = sqrt(Rw) * randn(1, k_bar);  % Measurement noise w_k ~ N(0, 4)
v = sqrt(Rv) * randn(1, k_bar);  % Process noise v_k ~ N(0, 1)

% Initial state for both controllers
x0 = mvnrnd(zeros(3,1), eye(3))';  % Initial state x0 ~ N(0, I)
x_LQR(:,1) = x0;
x_LQG(:,1) = x0;
x_hat_LQG(:,1) = [0; 0; 0];  % Initial estimate of the state (zero)

% Simulate both controllers
for k = 1:k_bar
    % LQR control (full state feedback)
    u_LQR(k) = -K{k} * x_LQR(:,k);
    % True system dynamics for LQR
    x_LQR(:,k+1) = A * x_LQR(:,k) + B_u * u_LQR(k) + B_v * v(k);
    
    % LQG control (output feedback with state estimation)
    y_LQG(k) = C * x_LQG(:,k) + w(k);  % Measurement with noise
    u_LQG(k) = -K{k} * x_hat_LQG(:,k);  % Control input based on estimated state
    % True system dynamics for LQG
    x_LQG(:,k+1) = A * x_LQG(:,k) + B_u * u_LQG(k) + B_v * v(k);
    % State estimation (Kalman filter update)
    x_hat_LQG(:,k+1) = A * x_hat_LQG(:,k) + B_u * u_LQG(k) + L{k} * (y_LQG(k) - C * x_hat_LQG(:,k));
end

% Plot the results
set(0, 'DefaultTextInterpreter', 'latex');  
set(0, 'DefaultLegendInterpreter', 'latex');  
figure;
% Plot state trajectories for LQR and LQG
plot(0:k_bar, x_LQR(3,:), "Color", 'black', 'DisplayName', '$(x_k)_3$ (LQR)');hold on;
plot(0:k_bar, x_LQG(3,:), 'b', 'DisplayName', '$(x_k)_3$ (LQG)');
xlabel('Time step $k$');
legend();
hold off;

figure;
% Plot state trajectories for LQG
plot(0:k_bar, x_LQG(1,:), 'b', 'DisplayName', '$(x_k)_1$ (LQG)');hold on;
plot(0:k_bar, x_hat_LQG(1,:), 'b--', 'DisplayName', '$(\hat{x}_k)_1$ (LQG)');
plot(1:k_bar, y_LQG(:), 'r', 'DisplayName', '$y_k$ (LQG)');
xlabel('Time step $k$');
legend();
hold off;

figure;
% Plot state trajectories for LQG
plot(0:k_bar, x_LQG(2,:), 'b', 'DisplayName', '$(x_k)_2$ (LQG)');hold on;
plot(0:k_bar, x_hat_LQG(2,:), 'b--', 'DisplayName', '$(\hat{x}_k)_2$ (LQG)');
xlabel('Time step $k$');
legend();
hold off;

figure;
% Plot state trajectories for LQG
plot(0:k_bar, x_LQG(3,:), 'b', 'DisplayName', '$(x_k)_3$ (LQG)');hold on;
plot(0:k_bar, x_hat_LQG(3,:), 'b--', 'DisplayName', '$(\hat{x}_k)_3$ (LQG)');
xlabel('Time step $k$');
legend();
hold off;