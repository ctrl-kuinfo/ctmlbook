# Author: Kenji Kashima
# Date  : 2023/03/12
# Note  : 

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import config


def lqr_control(A, B_u, Q, R, Qf, k_bar):
    """ Calculate the LQR feedback gains and cost-to-go matrices """
    P = [None] * (k_bar + 1)
    K = [None] * k_bar
    P[k_bar] = Qf
    
    # Backward recursion for finite-horizon LQR
    for k in range(k_bar - 1, -1, -1):
        K[k] = np.linalg.inv(R + B_u.T @ P[k + 1] @ B_u) @ (B_u.T @ P[k + 1] @ A)
        P[k] = Q + A.T @ P[k + 1] @ A - A.T @ P[k + 1] @ B_u @ K[k]
    
    return K, P

def kalman_filter(A, C, Qf, Rw, k_bar):
    """ Design the Kalman filter for state estimation """
    S = [None] * (k_bar + 1)
    L = [None] * k_bar
    S[k_bar] = Qf
    
    for k in range(k_bar - 1, -1, -1):
        L[k] = A @ S[k + 1] @ C.T @ np.linalg.inv(C @ S[k + 1] @ C.T + Rw)
        S[k] = (A @ S[k + 1] @ A.T 
                - A @ S[k + 1] @ C.T @ np.linalg.inv(C @ S[k + 1] @ C.T + Rw) @ C @ S[k + 1] @ A.T)
    
    return L, S

def simulate_lqr(A, B_u, B_v, C, K, k_bar, x0):
    """ Simulate the LQR controller """
    x_LQR = np.zeros((3, k_bar + 1))
    u_LQR = np.zeros(k_bar)
    x_LQR[:, 0] = x0

    Rv = 1  # Process noise covariance
    v = np.sqrt(Rv) * np.random.randn(k_bar)

    for k in range(k_bar):
        u_LQR[k] = -K[k] @ x_LQR[:, k]
        x_LQR[:, k + 1] = (A @ x_LQR[:, k].reshape(3,1) + B_u * u_LQR[k] + B_v * v[k]).flatten()
    
    return x_LQR, u_LQR

def simulate_lqg(A, B_u, B_v, C, K, L, k_bar, x0):
    """ Simulate the LQG controller """
    x_LQG = np.zeros((3, k_bar + 1))
    x_hat_LQG = np.zeros((3, k_bar + 1))
    u_LQG = np.zeros(k_bar)
    y_LQG = np.zeros(k_bar)

    Rw = 4  # Measurement noise covariance
    w = np.sqrt(Rw) * np.random.randn(k_bar)
    Rv = 1  # Process noise covariance
    v = np.sqrt(Rv) * np.random.randn(k_bar)

    x_LQG[:, 0] = x0
    x_hat_LQG[:, 0] = np.zeros(3)  # Initial estimate of the state (zero)

    for k in range(k_bar):
        # Output measurement with noise
        y_LQG[k] = C @ x_LQG[:, k] + w[k]

        # Control input using the estimated state
        u_LQG[k] = -K[k] @ x_hat_LQG[:, k]

        # True system dynamics for LQG
        x_LQG[:, k + 1] = (A @ x_LQG[:, k].reshape(3,1) + B_u * u_LQG[k] + B_v * v[k]).flatten()

        # State estimation (Kalman filter update)
        x_hat_LQG[:, k + 1] = (A @ x_hat_LQG[:, k].reshape(3,1) + B_u * u_LQG[k] + (L[k] @ (y_LQG[k] - C @ x_hat_LQG[:, k])).reshape(3,1)).flatten()

    return x_LQG, x_hat_LQG, u_LQG, y_LQG



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
    plt.plot(time_steps, x_LQG[0, :], 'b', label='$(x_k)_1$ (LQG)')
    plt.plot(time_steps, x_hat_LQG[0, :], 'b--', label='$(\hat{x}_k)_1$ (LQG)')
    plt.plot(time_steps[1:], y_LQG, 'r', label='$y_k$ (LQG)')
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
    plt.plot(time_steps, x_LQG[1, :], 'b', label='$(x_k)_2$ (LQG)')
    plt.plot(time_steps, x_hat_LQG[1, :], 'b--', label='$(\hat{x}_k)_2$ (LQG)')
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
    plt.plot(time_steps, x_LQG[2, :], 'b', label='$(x_k)_3$ (LQG)')
    plt.plot(time_steps, x_hat_LQG[2, :], 'b--', label='$(\hat{x}_k)_3$ (LQG)')
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