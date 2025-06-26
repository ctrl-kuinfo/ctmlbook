% Author: Kenji Kashima
% Date  : 2025/06/26

clear; close all;
rng(2); % random seed: Called once at the top level.

% --- Generate Figure 9.3(a) ---
disp('Generating Figure for sigma=2, iter_Gain=1 ...');
sigma_a = 2;
iter_Gain_a = 1;
iter_RLS_a = 50;
Npath_a = 20;
[x_norm_a, Kerr_a, Herr_a] = run_rl_simulation(sigma_a, iter_Gain_a, iter_RLS_a, Npath_a);
plot_rl_results(x_norm_a, Kerr_a, Herr_a, iter_Gain_a, '3a');

% --- Generate Figure 9.3(b) ---
disp('Generating Figure for sigma=10, iter_Gain=1 ...');
sigma_b = 10;
iter_Gain_b = 1;
iter_RLS_b = 50;
Npath_b = 20;
[x_norm_b, Kerr_b, Herr_b] = run_rl_simulation(sigma_b, iter_Gain_b, iter_RLS_b, Npath_b);
plot_rl_results(x_norm_b, Kerr_b, Herr_b, iter_Gain_b, '3b');

% --- Generate Figure 9.4 ---
disp('Generating Figure for sigma=10, iter_Gain=5 ...');
sigma_c = 10;
iter_Gain_c = 5;
iter_RLS_c = 50;
Npath_c = 20;
[x_norm_c, Kerr_c, Herr_c] = run_rl_simulation(sigma_c, iter_Gain_c, iter_RLS_c, Npath_c);
plot_rl_results(x_norm_c, Kerr_c, Herr_c, iter_Gain_c, '4');

disp('All figures generated.');


% --- Main Simulation Function (implements Algorithm 4 and 8) ---
function [x_norm_hist_p, Kerr_hist_p, Herr_hist_p] = run_rl_simulation(sigma, iter_Gain, iter_RLS, Npath)
    % Note: rng(2) is removed from here to allow for different random sequences per call.
    
    % --- Setup for LQR Problem (Problem 6.2.3) ---
    beta = 0.95; % discount rate (LQR for 1)
    A = [ 0.80    0.90    0.86;
          0.30    0.25    1.00;
          0.10    0.55    0.50];  % A-matrix
    B = [1; 0; 0];                % B-matrix
    [x_dim,u_dim] = size(B);      % state&input dim.
    % Dimension of the parameter vector for the quadratic Q-function (see Eq. 9.35)
    n_theta = (x_dim+u_dim)*(x_dim+u_dim+1)/2;     % size of theta
    E = eye(x_dim);         % cost x'Ex
    F = eye(u_dim);         % cost u'Fu
    % Optimal gain %
    [K_opt,~,~] = dlqr(A*sqrt(beta),B*sqrt(beta),E,F);
    K_opt = - K_opt;
    K_ini = [-4.1100,-11.7519,-19.2184];
    
    x_norm_hist_p = [];
    Herr_hist_p = [];
    Kerr_hist_p = [];
    
    for j = 1:Npath
        IsUnstable = 0;
        
        x_hist = [];
        Kerr_hist = [];
        Herr_hist = [];

        K = K_ini; % Current policy (gain)
        
        % === [Outer Loop] Policy Iteration (Algorithm 4) ===
        for k=1:iter_Gain
            % For comparison, calculate the true Q-function for the current K
            Pi = dlyap(sqrt(beta)*(A+B*K)' , [eye(x_dim,x_dim) K']*[eye(x_dim,x_dim);K]); % Solves the Lyapunov equation (9.22)
            Qopt = beta*[A B]'*Pi*[A B] + blkdiag(E,F); % Coefficient matrix of the Q-function (Eq. 9.23)
            Knext = - inv(Qopt(x_dim+1:x_dim+u_dim,x_dim+1:x_dim+u_dim))*Qopt(x_dim+1:x_dim+u_dim,1:x_dim);

            % === [Inner Loop] Policy Evaluation (TD Learning/RLS) (Core of Algorithm 8) ===
            theta = zeros(n_theta,1); % Initialization of Q-function parameter vector p-hat
            P = eye(n_theta)*10;   % RLS initialization
            
            x = zeros(3,1);   % Initial state of dynamics
            
            for i=1:iter_RLS
                x_hist = [x_hist, x];    
                
                % Determine the action by adding exploration noise (see Eq. 9.32)
                u = K*x + randn(u_dim,1)*sigma;   % tentative SF + exploration noise
                % Calculate the stage cost l_k (Eq. 9.1)
                cost = x'*E*x + u'*F*u;           % stage cost

                bar = phi(x,u);
                x = A*x + B*u;                  % One-step simulation
                u_next = K*x;
                barplus = phi(x,u_next);

                % Calculate the regressor vector for RLS (the {...} part in Eq. 9.34)
                phi_k = bar - beta*barplus;
                % Calculate the TD error (prediction error for RLS) (corresponds to Eq. 9.28)
                e = cost - phi_k'*theta;
                
                % Update parameter theta and covariance matrix P using the RLS algorithm (Algorithm 3)
                denom = 1 + phi_k'*P*phi_k;
                theta = theta + P*phi_k*e / denom;    
                P = P - (P*(phi_k*phi_k')*P)/ denom;
                
                % (For reference) Record the error of the current gain and Q-function
                Kerr_hist = [Kerr_hist, norm(K-K_opt,'fro')/norm(K_opt,'fro')];
                Herr_hist = [Herr_hist, norm(theta_to_H(theta,x_dim+u_dim)-Qopt,'fro')/norm(Qopt,'fro')];
            end
        
             % === Policy Improvement (Step 3 of Algorithm 5) ===
             % Reconstruct the Q-function matrix H from the estimated theta
             H = theta_to_H(theta,x_dim+u_dim);
             % Calculate the new gain K that minimizes the Q-function (Eq. 9.24)
             K = - inv(H(x_dim+1:x_dim+u_dim,x_dim+1:x_dim+u_dim))*H(x_dim+1:x_dim+u_dim,1:x_dim);
        
             % Stability check
             if max(abs(eig(A+B*K)))>1
                IsUnstable = 1;
                break;
             end
        end
        if IsUnstable==0
            x_norm_hist_p = [x_norm_hist_p;sqrt(sum(x_hist.^2,1))];
            Kerr_hist_p = [Kerr_hist_p;Kerr_hist];
            Herr_hist_p = [Herr_hist_p;Herr_hist];
        else 
           IsUnstable=0;
        end
    end
end


% --- Plotting Function ---
function plot_rl_results(x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain, label)
    grayColor = [.7 .7 .7];
    
    if isempty(Herr_hist_p)
        fprintf('Warning: No stable paths to plot for label %s.\n', label);
        return;
    end
    
    Effective_path = size(Herr_hist_p, 1);
    f = figure('Name', sprintf('Figure 9 - %s', label));
    
    % Plot state norm history
    subplot(2,1,1);
    hold on;
    plot(x_norm_hist_p','color' , grayColor);
    plot(mean(x_norm_hist_p,1),'LineWidth',2,'color',"black");
    ylabel('$\|x_k\|$','Interpreter','latex','Fontsize',15);
    total_steps = size(x_norm_hist_p, 2);
    xlim([1, total_steps]);
    if total_steps > 50
        xticks(0:50:total_steps);
    end
    grid on;
    
    % Plot Q-function error and Gain error
    subplot(2,1,2);
    hold on;
    
    % Left Y-axis (Q function error)
    yyaxis left;
    plot(Herr_hist_p','-','color' , grayColor);
    plot(mean(Herr_hist_p,1),'-','LineWidth',1.5,'color',"black");
    ylabel('Q function error','Fontsize',15);
    set(gca, 'YColor', 'black');
    ylim([0, 1.0]);
    
    % Right Y-axis (Gain error) - ONLY if iter_Gain > 1
    if iter_Gain > 1
        yyaxis right;
        plot(mean(Kerr_hist_p,1),'LineWidth',2,'color',"red");
        ylabel('Gain error','Fontsize',18);
        set(gca, 'YColor', 'red');
        max_y_right = max(mean(Kerr_hist_p,1));
        if max_y_right > 0
            ylim([0, max_y_right * 1.1]);
        end
    else
        % *** CORRECTED: Hide the right axis completely if not used ***
        ax = gca;
        ax.YAxis(2).Visible = 'off';
    end
    
    xlabel('$k$','Interpreter','latex','Fontsize',15);
    xlim([1, size(Herr_hist_p,2)]);
    grid on;
    
    if ~isempty(label)
        saveas(f, sprintf('Figure9_%s.pdf', label));
    end
end


% --- Helper Function Definitions ---
function phi_vector = phi(x,u)
    % Feature map phi(x,u) to represent the Q-function as a quadratic form (Eq. 9.35)
    H=[x;u]*[x;u]';
    n = size(H, 1);
    phi_vector = H(1,:)';
    for i = 2:n
        phi_vector = [phi_vector; H(i,i:n)'];
    end
end
function H = theta_to_H(theta,n)
    % Convert the parameter vector theta back to the matrix H
    H = theta(1:n)';
    k = n;
    for i = 2:n
        H = [ H ; [ zeros(1,i-1) theta(k+1:k+n-i+1)' ] ];
        k = k+n-i+1;
    end
    H = (H+H')/2;
end