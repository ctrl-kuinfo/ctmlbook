% Author: Kenji Kashima
% Date  : 2023/10/01

clear;close all; rng(3); % random seed

N_k = 50;
% Figure 11.2 (a) transition probability
P = [1/3    1/3      0      0;
       0    1/3    1/3      0;
       0    1/3    1/3    1/3;
     2/3      0    1/3    2/3];

p_stable = ones(4, 1) / 4;
for i = 1:100
    p_stable = P * p_stable;
end

% open loop stable state distribution
fprintf('p_100 = \n');
disp(p_stable);

% accumulated transition probability
[m,n] =size(P);
P_accum = zeros(m,n);
P_accum(1,:) = P(1,:);
for i = 2:m
    P_accum(i,:)= P_accum(i-1,:)+P(i,:);
end


p_list = zeros(1,N_k); 
p_list(1)=4; % start at 4
for i=2:N_k
    u = rand;
    T = P_accum(:,p_list(i-1)); %transition probability
    for j = 1:m
        if u <= T(j)
             p_list(i) = j;
             break
        end
    end
end

figure('Name','Figure10.2(b)'); hold on; grid on;
stairs(p_list);
xlim([0,N_k]);
ylim([0.8,4.2]);
xlabel('$k$','Fontsize',16,'Interpreter','latex')
title('$P^0$','Fontsize',16,'Interpreter','latex')








