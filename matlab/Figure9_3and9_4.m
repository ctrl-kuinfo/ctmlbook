% Author: Kenji Kashima
% Date  : 2024/10/16

clear;close all; rng(2); % random seed
beta = 0.95; % discount rate (LQR for 1)
A = [ 0.80    0.90    0.86;
      0.30    0.25    1.00;
      0.10    0.55    0.50];  % A-matrix
B = [1; 0; 0];                % B-matrix

[x_dim,u_dim] = size(B);      % state&input dim.
n_theta = (x_dim+u_dim)*(x_dim+u_dim+1)/2;     % size of theta
% theta is the parameters for
% x1^2 x1x2 x1x3 x1u1
%      x2^2 x2x3 x2u1
%           x3^2 x3u1
%                u1^2

E = eye(x_dim);         % cost x'Ex
F = eye(u_dim);         % cost u'Fu

sigma = 10;         % Probing noise scale

% Optimal gain %
[K_opt,~,~] = dlqr(A*sqrt(beta),B*sqrt(beta),E,F);
K_opt = - K_opt;
K_ini = [-4.1100,-11.7519,-19.2184];
IsUnstable = 0;

x_norm_hist_p = [];    % history of x
Herr_hist_p = []; % history of \|H-Pi\|_F
Kerr_hist_p = []; % history of \|K-K_opt\|_F

iter_Gain = 5;    % Gain-update iteration
iter_RLS =50;     % RLS iteration
Npath = 20;

for j = 1:Npath
    x_hist = [];    % history of x
    Kerr_hist = []; % history of \|U-Uopt\|_F
    Herr_hist = []; % history of \|H-Pi\|_F
    
    theta = zeros(n_theta,1); 
    
    K = K_ini;
    x = zeros(3,1);   % Initial state of dynamics
    for k=1:iter_Gain
        Pi = dlyap(sqrt(beta)*(A+B*K)' , [eye(x_dim,x_dim) K']*[eye(x_dim,x_dim);K]);
        Qopt = beta*[A B]'*Pi*[A B] + blkdiag(E,F);
        Knext = - inv(Qopt(x_dim+1:x_dim+u_dim,x_dim+1:x_dim+u_dim))*Qopt(x_dim+1:x_dim+u_dim,1:x_dim);
    
        theta = zeros(n_theta,1);
        P = eye(n_theta)*10;   % RLS initialization
        for i=1:iter_RLS
            x_hist = [x_hist, x];    
            Kerr_hist = [Kerr_hist, norm(K-K_opt,'fro')/norm(K_opt,'fro')];
            Herr_hist = [Herr_hist, norm(theta_to_H(theta,x_dim+u_dim)-Qopt,'fro')/norm(Qopt,'fro')];
           
            u = K*x + randn(u_dim,1)*sigma;   % tentative SF + exploration noise
            cost = x'*E*x + u'*F*u;           % stage cost
            
            bar = phi(x,u);
            x = A*x + B*u;                  % One-step simulation
            u = K*x;
            barplus = phi(x,u);
            
            phi_k = bar - beta*barplus;
            e = cost - phi_k'*theta;              % around (9.37)
            denom = 1 + phi_k'*P*phi_k;
            theta = theta + P*phi_k*e / denom;    
            P = P - (P*(phi_k*phi_k')*P)/ denom;     
        end
    
         H = theta_to_H(theta,x_dim+u_dim);
         K = - inv(H(x_dim+1:x_dim+u_dim,x_dim+1:x_dim+u_dim))*H(x_dim+1:x_dim+u_dim,1:x_dim);
         Kerr = norm(K-Knext,'fro');   
    
         if max(abs(eig(A+B*K)))>1
            IsUnstable = 1;
            break;
         end
    end

    if IsUnstable==0
        x_norm_hist_p = [x_norm_hist_p;sqrt(diag(x_hist' * x_hist)' )];    % history of x
        Kerr_hist_p = [Kerr_hist_p;Kerr_hist];    % history of U
        Herr_hist_p = [Herr_hist_p;Herr_hist];    % history of \|H-Pi\|_F
    else 
       IsUnstable=0;
    end

end

grayColor = [.7 .7 .7];
Effective_path = size(Herr_hist_p)*[1;0];

f = figure;
subplot(2,1,1);
hold on;
plot(x_norm_hist_p','color' , grayColor);
plot(x_norm_hist_p'*ones(Effective_path,1)/Effective_path,'LineWidth',2,'color',"black");

subplot(2,1,2);
hold on;
yyaxis left;
plot(Herr_hist_p','-','color' , grayColor);
plot(Herr_hist_p'*ones(Effective_path,1)/Effective_path,'-','LineWidth',1.5,'color',"black");

ylabel('Q function error','Fontsize',15);

yyaxis right;
plot(Kerr_hist_p'*ones(Effective_path,1)/Effective_path,'LineWidth',2,'color',"red");
ylabel('Gain error','Fontsize',18);
xlabel('$k$','Interpreter','latex','Fontsize',15);


function phi_vector = phi(x,u)
    % Convert matrix H to vector theta by stacking the upper triangular elements 
    H=[x;u]*[x;u]';
    n = size(H, 1);  % Assuming H is a square matrix
    phi_vector = H(1,:)';  % First row
    for i = 2:n
        phi_vector = [phi_vector; H(i,i:n)'];  % Stack rows i to n
    end
end

function H = theta_to_H(theta,n)
    H = theta(1:n)';
    k = n;
    for i = 2:n
        H = [ H ; [ zeros(1,i-1) theta(k+1:k+n-i+1)' ] ];
        k = k+n-i+1;
    end
    H = (H+H')/2;
end

