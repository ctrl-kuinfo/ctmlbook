% Author: Kenji Kashima
% Date  : 2025/04/01
% Note  : Optimization Toolbox (quadprog) 

clear; close all; rng(1);   % random seed

% ========= ground truth =========
phi    = @(x) [1 ; x ; x^2 ; x^3 ; x^4 ; x^5 ; x^6 ; x^7 ; x^8 ; x^9];
f_true = @(x) 2*sin(5*x);

n_f = numel(phi(0));        % feature dimension

% ========= データ生成 =========
n_data = 30;
x_s = rand(n_data,1);
X = zeros(n_data, n_f); 
y = zeros(n_data,1);
for i = 1:n_data
    X(i,:) = phi(x_s(i)).';
    y(i)   = f_true(x_s(i)) + randn;   % ノイズ付き
end

sigma_sq = 0.01;            

% ========= 1) Naive =========
% minimize ||X*theta - y||_2
theta_naive = X \ y;

% ========= 2) Ridge =========
% minimize (1/n)||X*theta - y||_2^2 + λ||theta||_2^2
% → θ = (X'X + nλ I)^{-1} X'y
n = size(X,1);
theta_ridge = (X.'*X + n*sigma_sq*eye(n_f)) \ (X.'*y);

% ========= 3) Lasso（quadprog）=========
% minimize (1/n)||X*theta - y||_2^2 + λ||theta||_1
% 補助変数 t >= 0 を導入して QP 化：
%   z = [theta; t]
%   (1/2) z'H z + f'z を最小化
%     H = blkdiag( (2/n)X'X, 0 ),
%     f = [ -(2/n)X'y ; λ * 1 ]
%   制約: -t <= theta <= t  ⇔  [ I  -I; -I  -I ] z <= 0, かつ t >= 0
H = blkdiag( (2/n)*(X.'*X), zeros(n_f) );          % (2n_f x 2n_f)
f = [ -(2/n)*(X.'*y) ; sigma_sq*ones(n_f,1) ];     % (2n_f x 1)

A = [ eye(n_f), -eye(n_f) ;     %  theta - t <= 0  (theta <= t)
     -eye(n_f), -eye(n_f) ];    % -theta - t <= 0  (theta >= -t)
b = zeros(2*n_f,1);

lb = [ -inf(n_f,1) ; zeros(n_f,1) ];   % theta: 自由,  t: >= 0
ub = [];                                % 上限なし

opts = optimoptions('quadprog','Display','off','Algorithm','interior-point-convex');
z = quadprog(H, f, A, b, [], [], lb, ub, [], opts);
theta_lasso = z(1:n_f);

% ========= 予測曲線の作図 =========
n_x = 100;
x_grid = linspace(0,1,n_x);
NAIVE_dat = zeros(1,n_x);
LASSO_dat = zeros(1,n_x);
RIDGE_dat = zeros(1,n_x);
f_real    = zeros(1,n_x);
for i = 1:n_x
    ph = phi(x_grid(i));
    NAIVE_dat(i) = theta_naive' * ph;
    LASSO_dat(i) = theta_lasso' * ph;
    RIDGE_dat(i) = theta_ridge' * ph;
    f_real(i)    = f_true(x_grid(i));
end

% --- Figure 7.4(a) ---
figure('Name','Figure 7.4(a)'); hold on; grid on;
scatter(x_s, y, 25, 'MarkerEdgeColor',[0.2 0.2 0.2], 'DisplayName','samples');
plot(x_grid, NAIVE_dat, 'k','LineWidth',2, 'DisplayName','Least Squares');
plot(x_grid, RIDGE_dat, 'b','LineWidth',2, 'DisplayName','Ridge');
plot(x_grid, LASSO_dat, 'r','LineWidth',2, 'DisplayName','Lasso');
plot(x_grid, f_real,    'm--','LineWidth',1.5,'DisplayName','True 2sin(5x)');
axis([0 1 -5 5]);
xlabel('x','FontSize',14);
ylabel('f(x)','FontSize',14);
legend('Location','best');
set(gca,'FontSize',12);
movegui('northeast');

% --- Figure 7.4(b) ---
figure('Name','Figure 7.4(b)'); hold on; grid on;
plot(1:n_f, abs(theta_ridge), 'b.','MarkerSize',16,'DisplayName','Ridge |θ|');
plot(1:n_f, abs(theta_lasso), 'ro','MarkerSize',5, 'DisplayName','Lasso |θ|');
xlim([1 n_f]); xticks(1:n_f);
xlabel('i','FontSize',14);
ylabel('|coeff|','FontSize',14);
legend('Location','best');
set(gca,'FontSize',12);
movegui('northwest');
