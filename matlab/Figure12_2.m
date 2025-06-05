% Author: Kenji Kashima
% Date  : 2025/06/05 
% Note  : Statistics and Machine Learning Toolbox is NOT required

clear; close all; rng(3); % random seed

N_x = 100;                         % Number of x points
x_list = linspace(0,1,N_x);        % x values on [0,1]

% Define gaussian_kernel
gaussian_kernel = @(x, y, c) exp(- (x - y').^2 / c^2);  % eq 13.7 & 13.8

% Original function in Example 13.1.7: mu(x) = x
mu_x = @(x) x;
y_list = mu_x(x_list');

% Loop over c = 0.1 (Figure 12.2(a)) and c = 1.0 (Figure 12.2(b))
for idx = 1:2
    if idx == 1
        c = 0.1; 
        label = 'a';
    else
        c = 1.0;
        label = 'b';
    end

    % Compute kernel matrix K = gaussian_kernel(x_list, x_list, c)
    K = gaussian_kernel(x_list, x_list, c);

    % Cholesky decomposition (with jitter if needed)
    jitter = 1e-8;
    while true
        [L, p] = chol(K + jitter * eye(N_x), 'lower');
        if p == 0
            break;
        else
            jitter = jitter * 10;
        end
    end

    % Plot Figure 12.2(a) or (b)
    figure('Name', sprintf('Figure12.2(%s)', label)); hold on; grid on;
    plot(x_list, y_list, 'k', 'LineWidth', 2);  % μ(x)
    for i = 1:5
        z = randn(N_x, 1);              % standard normal vector
        sample = y_list + L * z;        % multivariate normal sample
        plot(x_list, sample);
    end

    % Fill ±1σ region
    kxx = sqrt(diag(K));
    x_fill = [x_list, fliplr(x_list)]; 
    y_fill = [y_list - kxx; flipud(y_list + kxx)]; 
    fill(x_fill, y_fill, 'b', ...
         'FaceAlpha', 0.3, 'EdgeColor', 'none');

    xlabel('$x$', 'Interpreter', 'latex', 'Fontsize', 18);
    ylabel('$y$', 'Interpreter', 'latex', 'Fontsize', 18);
    legend('$\mu(x)$', 'Interpreter', 'latex', 'Fontsize', 16, 'Location', 'northwest');
    xlim([0 1]);
    ylim([-3 3]);

    set(gca, 'Fontsize', 12);
    hold off;
end