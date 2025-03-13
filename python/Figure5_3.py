# Author: Kenji Kashima
# Date  : 2023/03/12
# Note  : 

import numpy as np
import matplotlib.pyplot as plt
from module_lqg import lqr_control,kalman_filter,simulate_lqg,simulate_lqr
import sys
sys.path.append("./")
import config
np.random.seed(1)  # Random seed
x_max = 6

def figure5_3a(x_LQR, x_LQG, k_bar):
    """ Plot state trajectories for LQR and LQG controllers """
    figsize = config.global_config(type=0)

    time_steps = np.arange(0, k_bar + 1)

    plt.figure(figsize=figsize)
    plt.plot(time_steps, x_LQR[2, :], 'k', label='$(x_k)_3$ (LQR)')
    plt.plot(time_steps, x_LQG[2, :], 'b', label='$(x_k)_3$ (LQG)')
    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.savefig("./figures/Figure5_3a.pdf")
    plt.show()

def figure5_3b(x_LQG, x_hat_LQG, y_LQG, k_bar):
    """ Plot state $(x_k)_1$ and its estimate with output measurement """
    figsize = config.global_config(type=0)

    time_steps = np.arange(0, k_bar + 1)

    plt.figure(figsize=figsize)
    plt.plot(time_steps, x_LQG[0, :], 'b', label='True state $(x_k)_1$')
    plt.plot(time_steps[1:], y_LQG, 'r', label='Measurements $y_k$')
    plt.plot(time_steps, x_hat_LQG[0, :], 'b--', label='Estimated $(\hat{x}_k)_1$')
    plt.xlabel('$k$')
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.savefig("./figures/Figure5_3b.pdf")
    plt.show()

def figure5_3c(x_LQG, x_hat_LQG, k_bar):
    """ Plot state $(x_k)_2$ and its estimate """
    figsize = config.global_config(type=0)

    time_steps = np.arange(0, k_bar + 1)

    plt.figure(figsize=figsize)
    plt.plot(time_steps, x_LQG[1, :], 'b', label='True state $(x_k)_2$')
    plt.plot(time_steps, x_hat_LQG[1, :], 'b--', label='Estimated $(\hat{x}_k)_2$')
    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.savefig("./figures/Figure5_3c.pdf")
    plt.show()

def figure5_3d(x_LQG, x_hat_LQG, k_bar):
    """ Plot state $(x_k)_3$ and its estimate """
    figsize = config.global_config(type=0)

    time_steps = np.arange(0, k_bar + 1)

    plt.figure(figsize=figsize)
    plt.plot(time_steps, x_LQG[2, :], 'b', label='True state $(x_k)_3$')
    plt.plot(time_steps, x_hat_LQG[2, :], 'b--', label='Estimated $(\hat{x}_k)_3$')
    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0,k_bar])
    plt.ylim([-x_max,x_max])
    plt.savefig("./figures/Figure5_3d.pdf")
    plt.show()

# Define system matrices
A = np.array([[0.40, 0.37, 0.09],
              [0.52, 0.66, 0.15],
              [0.21, 0.66, 0.04]])  
B_u = np.array([[0], [1], [0]])  # Control input matrix
B_v = np.array([[1], [0], [0]])  # Disturbance input matrix (noise)
C = np.array([[1, 0, 0]])        # Output matrix

# LQR parameters
Q = np.diag([0, 0, 1])           # State cost matrix
R = 1                             # Control input cost
Qf = np.eye(3)                   # Final state cost
k_bar = 60                        # Total time steps

# Noise properties
Rv = 1                            # Process noise covariance (v_k ~ N(0, 1))
Rw = 4                            # Measurement noise covariance (w_k ~ N(0, 4))

# Compute LQR feedback gains
K, P = lqr_control(A, B_u, Q, R, Qf, k_bar)

# Compute Kalman gains
L, S = kalman_filter(A, C, Qf, Rw, k_bar)

# Initial state
x0 = np.random.multivariate_normal(np.zeros(3), np.eye(3))  # Initial state x0 ~ N(0, I)

# Simulate LQR and LQG controllers
x_LQR, u_LQR = simulate_lqr(A, B_u, B_v, C, K, k_bar, x0)
x_LQG, x_hat_LQG, u_LQG, y_LQG = simulate_lqg(A, B_u, B_v, C, K, L, k_bar, x0)

if __name__ == '__main__':
    # Plot the results
    figure5_3a(x_LQR,x_LQG,k_bar)
    figure5_3b(x_LQG, x_hat_LQG, y_LQG, k_bar)
    figure5_3c(x_LQG, x_hat_LQG, k_bar)
    figure5_3d(x_LQG, x_hat_LQG, k_bar)