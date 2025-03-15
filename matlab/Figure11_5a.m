% Author: Kenji Kashima
% Date  : 2023/11/05

figure('Name','Figure11.5(a)'); hold on; grid on;
x = -5:0.01:5;
y = x.^2/2-x.*cos(x*10)/20+sin(x*10)/20;
plot(x,y);
y = x+x.*sin(x*10)/2;
plot(x,y);
y = x+1/2;
plot(x,y)
y=x.*sin(x*10)/2 -1/2;
plot(x,y)

grid on;
xlabel('$x$','Interpreter','latex','Fontsize',18);
ylabel('$x_k$','Interpreter','latex','Fontsize',18);
legend('$L(x)$','$\nabla L(x)$','$\nabla L_1(x)$','$\nabla L_2(x)$','Interpreter','latex','Fontsize',10)

