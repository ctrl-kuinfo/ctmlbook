% Author: Kenji Kashima
% Date  : 2025/06/05 (updated to match Python version)

clear; close all; rng(3);  % random seed

N_x = 100;   % number of x in the x_bar for depicting the original function
s_bar_list = [10, 50];  % use s_bar=10 for Figure12.3(a), s_bar=50 for Figure12.3(b)

% original function in examples 13.1.7 and 13.1.9
mu_x = @(x) x;
fn_x = @(x) sin(4*pi*x);

% Gaussian‐kernel function, matching Python version
function K = gaussian_kernel(x, y, c)
    % x: (n_x × 1), y: (n_y × 1)
    % returns (n_x × n_y) matrix with entries exp(−(x_i − y_j)^2 / c^2)
    K = exp(-(x - y').^2 / c^2);
end

for idx = 1:2
    s_bar = s_bar_list(idx);
    
    x_bar = linspace(0, 1, N_x)';       % (N_x × 1)
    x_sample = rand(s_bar, 1) .* rand(s_bar, 1);  % (s_bar × 1), random samples
    
    % kernel bandwidth
    c = 0.1;
    
    % compute K (s_bar × s_bar) and kx (s_bar × N_x)
    K = gaussian_kernel(x_sample, x_sample, c);         % eq 13.24
    kx = gaussian_kernel(x_sample, x_bar, c);           % eq 13.16
    
    mu_list = mu_x(x_bar);            % (N_x × 1)
    fn_list = fn_x(x_bar);            % (N_x × 1)
    
    % generate noisy observations: y_sample = f(x_sample) + Gaussian noise
    sigma = 0.1;
    e_s = sigma * randn(s_bar, 1);
    y_sample = fn_x(x_sample) + e_s;  % eq 13.22

    % compute posterior mean and variance
    y_mean = mu_x(x_sample);  % (s_bar × 1)
    % posterior mean: μ(x|D) = μ(x_bar) + kx' * (K + σ^2 I)^{-1} * (y_sample − μ(x_sample))
    inv_term = (K + sigma^2 * eye(s_bar)) \ (y_sample - y_mean);  % (s_bar × 1)
    mean_post = mu_x(x_bar) + kx' * inv_term;                     % (N_x × 1)
    % posterior variance diagonal: var(x) = diag(kxx − kx' * (K + σ^2 I)^{-1} * kx)
    kxx = gaussian_kernel(x_bar, x_bar, c);  % (N_x × N_x)
    var_matrix = kxx - kx' * ((K + sigma^2 * eye(s_bar)) \ kx);  % (N_x × N_x)
    vm = sqrt(diag(var_matrix));  % (N_x × 1)

    % prepare fill region
    x_fill = [x_bar', fliplr(x_bar')];                  % (1 × 2N_x)
    y_fill = [mean_post - vm; flipud(mean_post + vm)];  % (2N_x × 1)

    % plot
    figure('Name', sprintf('Figure12.3(%c)', char('a'+idx-1))); hold on; grid on;
    % plot μ(x)
    plot(x_bar, mu_list, 'r', 'LineWidth', 1);
    % plot f(x) = sin(4πx)
    plot(x_bar, fn_list, 'k--', 'LineWidth', 1);
    % scatter samples
    scatter(x_sample, y_sample, 'b', 'filled');
    % fill ±1‐sigma band
    fill(x_fill, y_fill, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    % plot posterior mean μ(x|D) as blue dashed line
    plot(x_bar, mean_post, 'b--', 'LineWidth', 1);

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 18);
    legend(...
        {'$\mu(x)$', '$\sin(4\pi x)$', '${\rm samples}\;\;y_s$', '', '$\mu(x|\mathcal D)$'}, ...
        'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');
    xlim([0, 1]);
    ylim([-1.5, 3.0]);
    hold off;
end
