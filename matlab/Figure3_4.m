% Author: Kenji Kashima
% Date  : 2025/04/01
% Note  : requires DSP Toolbox or Signal Processing Toolbox

clear;close all; rng(1); % random seed


% Figure 3.4(a)
[b,a] = cheby2(3,40,1/pi); % Chebyshev Type II filter design
[data,omega] = freqz(b,a); % Frequency response of digital filter
figure('Name','Figure 3.4(a)'); hold on; grid on;
plot(omega,abs(data),'linewidth',1)
plot([0,0.25,0.6,pi],[1,1,0,0],'linewidth',2,'linestyle','--')

xlim([0,pi])
xlabel('$\varpi$','Fontsize',18,'Interpreter','latex')
legend('frequency weight','prior information','Fontsize',16)
movegui('northwest')


% Figure 3.4(b)
freq_model = tf(b,a,1);
n_k = 200; % number of k
v_k = randn(n_k+1, 1);    % random input from [0,1]
y_k = lsim(freq_model,v_k); % Plot simulated time response of dynamic system 
                         % to arbitrary inputs; simulated response data

figure('Name','Figure 3.4(b)'); hold on; grid on;
stairs(v_k)
stairs(y_k,'linewidth',1)
xlim([0,200])
ylim([-2,2])
xlabel('$k$','Fontsize',18,'Interpreter','latex')
legend('white','colored','Fontsize',16)
movegui('northeast')
