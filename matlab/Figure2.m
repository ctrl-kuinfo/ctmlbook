% Author: Kenji Kashima
% Date  : 2025/04/01

figure2_1a
figure2_1b

function figure2_1a()
    % Set up the x values from -10 to 10 with 1000 points
    x = linspace(-10, 10, 1000);

    % Probability density function for N(-2, 1)
    pdf_x1 = normpdf(x, -2, 1);  % N(-2, 1)

    % Probability density function for N(2, 1)
    pdf_x2 = normpdf(x, 2, 1);   % N(2, 1)

    % Probability density function for (x1 + x2) ~ N(0, sqrt(2))
    pdf_x1_plus_x2 = normpdf(x, 0, sqrt(2)); 

    % Mixture of two normal distributions
    pdf_mixture = 0.5 * normpdf(x, -2, 1) + 0.5 * normpdf(x, 2, 1); 

    % Laplace distribution with parameters (0, 1)
    pdf_x3 = laplace_pdf(x, 0, 1); 

    % Create a figure with specified size
    figure('Position', [100, 100, 800, 800]);

    % Plot the probability density functions
    plot(x, pdf_x1, 'b', 'DisplayName', '$x_1 \sim \mathcal{N}(-2, 1)$'); hold on;
    plot(x, pdf_x2, 'Color', [1, 0.647, 0], 'DisplayName', '$x_2 \sim \mathcal{N}(2, 1)$');  % Orange
    plot(x, pdf_x1_plus_x2, 'g', 'DisplayName', '$(x_1+x_2) \sim \mathcal{N}(0, 2)$'); 
    plot(x, pdf_mixture, 'r', 'DisplayName', '$0.5\mathcal{N}(-2, 1) + 0.5\mathcal{N}(2, 1)$'); 
    plot(x, pdf_x3, 'm', 'DisplayName', '$x_3 \sim \rm{Lap}(0, 1)$');  % Purple

    % Add legend and labels
    legend('show','Interpreter','latex');
    xlabel('x');
    xlim([-5, 5]);
    grid on;
   
    hold off;  % Release hold on the current plot
end

% Custom function for Laplace probability density function
function y = laplace_pdf(x, mu, b)
    % Laplace PDF calculation
    y = (1/(2*b)) * exp(-abs(x - mu) / b);
end


function figure2_1b()
    % Set up the x values from 0 to 10 with 1000 points
    x = linspace(0, 10, 1000);

    % Probability density function for N(2, 1)
    pdf_x2 = normpdf(x, 2, 1);  % N(2, 1)

    % Laplace distribution with parameters (0, 1)
    pdf_x3 = laplace_pdf(x, 0, 1); 

    % Create a figure with specified size
    figure('Position', [100, 100, 800, 800]);

    % Plot the probability density functions
    plot(x, pdf_x2, 'Color', [1, 0.647, 0], 'DisplayName', '$\mathcal{N}(2, 1)$'); hold on; % Orange
    plot(x, pdf_x3, 'm', 'DisplayName', '$\rm{Lap}(0, 1)$');  % Purple

    % Add legend and labels
    legend('show');
    xlabel('|x|');
    xlim([0, 10]);
    ylim([1e-12, 1]);
    set(gca, 'YScale', 'log');  % Set y-axis to logarithmic scale
    grid on;
    
    hold off;  % Release hold on the current plot
end


