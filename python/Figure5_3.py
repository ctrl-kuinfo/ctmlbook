# Author: Kenji Kashima
# Date  : 2025/06/21
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import config
np.random.seed(1)  # Random seed
x_max = 6

def lqr_control(A, B_u, Q, R, Qf, k_bar):
    """
    Finite-horizon discrete-time LQR controller
    Calculates feedback gains K and cost-to-go matrices Sigma for k=0...k_bar

    Parameters
    ----------
    A, B_u : system matrices (x_{k+1} = A x_k + B_u u_k)
    Q, R   : stage cost matrices
    Qf     : terminal cost matrix
    k_bar  : horizon length (number of steps)

    Returns
    -------
    K     : list of feedback gain matrices, length k_bar
    Sigma : list of cost-to-go matrices, length k_bar+1
    """
    # Initialize lists to store Sigma_k and K_k
    Sigma = [None] * (k_bar + 1)  # Sigma[k] = cost-to-go at step k
    K = [None] * k_bar            # K[k]    = feedback gain at step k

    # Terminal cost
    Sigma[k_bar] = Qf

    # Backward Riccati recursion
    for k in range(k_bar - 1, -1, -1):
        # Compute R̃_k = R + B_u^T Sigma[k+1] B_u
        R_tilde = R + B_u.T @ Sigma[k + 1] @ B_u
        # Compute S̃_k = A^T Sigma[k+1] B_u
        S_tilde = A.T @ Sigma[k + 1] @ B_u
        # Compute Q̃_k = Q + A^T Sigma[k+1] A
        Q_tilde = Q + A.T @ Sigma[k + 1] @ A

        # Feedback gain K[k] = R̃_k^{-1} S̃_k^T
        K[k] = np.linalg.solve(R_tilde, S_tilde.T)

        # Update cost-to-go Sigma[k]
        Sigma[k] = Q_tilde - S_tilde @ np.linalg.solve(R_tilde, S_tilde.T)

    return K, Sigma


def simulate_lq_control(A, B_u, B_v, C, Sigma, K, k_bar, x0, mode, Rw=4.0, Rv=1.0, v=None):
    """
    Simulate LQG controller using Algorithm 1 order
    Records true state, state estimates, inputs, measurements, and covariance

    Parameters
    ----------
    A, B_u, B_v, C : system matrices
    Sigma          : Inistial covariance
    K              : list of finite-horizon LQR gains K[k]
    k_bar          : horizon length
    x0             : initial state vector
    mode           : 'lqr', 'lqg_kalman', or 'lqg_pred'
    Rw, Rv         : measurement and process noise covariances
    v              : optional pre-generated process noise sequence

    Returns
    -------
    x_true : true state trajectories (nx x (k_bar+1))
    x_hat  : state estimates        (nx x (k_bar+1))
    u      : control inputs         (k_bar,)
    y      : measurements           (k_bar,)
    Sigmas : covariance matrices    (nx x nx x (k_bar+1))
    x_check: corrected state estimates        (nx x (k_bar))
    Sigmac : corrected covariance matrices    (nx x nx x (k_bar))
    """
    nx = A.shape[0]

    # Generate noise sequences if not provided
    w = np.sqrt(Rw) * np.random.randn(k_bar)  # measurement noise w_k ~ N(0, Rw)
    if v is None:
        v = np.sqrt(Rv) * np.random.randn(k_bar)  # process noise v_k ~ N(0, Rv)

    # Allocate containers
    x_true  = np.zeros((nx, k_bar + 1))
    x_hat   = np.zeros((nx, k_bar + 1))
    Sigmas  = np.zeros((nx, nx, k_bar + 1))
    x_check = np.zeros((nx, k_bar))
    Sigmac  = np.zeros((nx, nx, k_bar))
    u       = np.zeros(k_bar)
    y       = np.zeros(k_bar)

    # Initial conditions
    x_true[:, 0] = x0
    x_hat[:, 0]  = np.zeros(nx)
    Sigmas[:, :, 0] = Sigma

    # Main loop over time steps
    for k in range(k_bar):
        # If perfect state measurement (LQR)
        if mode == 'lqr':
            u[k] = -K[k] @ x_true[:, k]
        else:
            # For prediction mode, use prior estimate
            if mode == 'lqg_pred':
                u[k] = -K[k] @ x_hat[:, k]

            # 1) Receive measurement y_k
            y[k] = C @ x_true[:, k] + w[k]

            # 2) Compute Kalman gain components: M̃_k, Ľ_k, Ȟ_k
            M_tilde  = C @ Sigma @ C.T + Rw      # innovation covariance
            L_check  = Sigma @ C.T               # cross-covariance
            H_check  = L_check @ np.linalg.inv(M_tilde)  # Kalman gain

            # 3) Posterior update of state estimate x̌_k
            innov    = y[k] - C @ x_hat[:, k]
            x_check[:, k]  = x_hat[:, k] + H_check.flatten() * innov

            # 4) Posterior covariance Σ̌_k
            Sigma_check = Sigma - L_check @ np.linalg.inv(M_tilde) @ L_check.T
            Sigmac[:, :, k] = Sigma_check

            # Choose input using updated estimate for LQG
            if mode == 'lqg_kalman':
                u[k] = -K[k] @ x_check[:, k]

            # 5) Time update (prior for next step)
            x_hat[:, k+1] = (A @ x_check[:,k].reshape(nx,1) + B_u * u[k]).flatten()
            Sigma = A @ Sigma_check @ A.T + Rv * np.eye(nx)

        # 6) Propagate true system
        x_true[:, k+1] = (A @ x_true[:, k].reshape(nx,1) + B_u * u[k] + B_v * v[k]).flatten()
        Sigmas[:, :, k+1] = Sigma
        
    return x_true, x_hat, u, y, Sigmas, x_check, Sigmac


def figure5_3a(x_LQR, x_true, Sigmas, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    plt.figure(figsize=figsize)
    plt.plot(t, x_LQR[2], 'k', label='$(x_k)_3$ (LQR)')
    plt.plot(t, x_true[2], 'b', label='$(x_k)_3$ (LQG)')
    sd = np.sqrt(Sigmas[2, 2])
    # plt.fill_between(t, x_true[2] - sd, x_true[2] + sd, color='blue', alpha=0.2)

    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, k_bar])
    plt.ylim([-x_max, x_max])

    plt.tight_layout()
    plt.savefig("./figures/Figure5_3a.pdf")
    plt.show()

def figure5_3b(x_true, x_hat, y, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    t_c = np.arange(k_bar)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[0], 'b', label='True $(x_k)_1$')
    plt.plot(t_c, y, 'r', label='Measurements $y_k$')
    plt.plot(t, x_hat[0], 'b--', label='Estimate $(\hat x_k)_1$')
    sd = np.sqrt(Sigmas[0, 0])
    plt.fill_between(t, x_hat[0] - sd, x_hat[0] + sd, color='blue', alpha=0.2)

    # #  Plot corrected estimate
    # sd_c = np.sqrt(Sigmac[0, 0])
    # plt.plot(t_c, x_check[0], 'r--', label='Corrected estimate $(\check x_k)_1$')
    # plt.fill_between(t_c, x_check[0] - sd_c, x_check[0] + sd_c, color='red', alpha=0.2)

    plt.xlabel('$k$')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, k_bar])
    plt.ylim([-x_max, x_max])

    plt.tight_layout()
    plt.savefig("./figures/Figure5_3b.pdf")
    plt.show()

def figure5_3c(x_true, x_hat, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    t_c = np.arange(k_bar)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[1], 'b', label='True $(x_k)_2$')
    plt.plot(t, x_hat[1], 'b--', label='Estimate $(\hat x_k)_2$')
    sd = np.sqrt(Sigmas[1, 1])
    plt.fill_between(t, x_hat[1] - sd, x_hat[1] + sd, color='blue', alpha=0.2)

    ## Plot corrected estimate
    # sd_c = np.sqrt(Sigmac[1, 1])
    # plt.plot(t_c, x_check[1], 'r', label='Corrected estimate $(\check x_k)_2$')
    # plt.fill_between(t_c, x_check[1] - sd_c, x_check[1] + sd_c, color='red', alpha=0.2)

    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, k_bar])
    plt.ylim([-x_max, x_max])

    plt.tight_layout()
    plt.savefig("./figures/Figure5_3c.pdf")
    plt.show()

def figure5_3d(x_true, x_hat, Sigmas, x_check, Sigmac, k_bar):
    figsize = config.global_config(type=0)
    t = np.arange(k_bar + 1)
    t_c = np.arange(k_bar)
    plt.figure(figsize=figsize)
    plt.plot(t, x_true[2], 'b', label='True $(x_k)_3$')
    plt.plot(t, x_hat[2], 'b--', label='Estimate $(\hat x_k)_3$')
    sd = np.sqrt(Sigmas[2, 2])
    plt.fill_between(t, x_hat[2] - sd, x_hat[2] + sd, color='blue', alpha=0.2)

    ## Plot corrected estimate
    # sd_c = np.sqrt(Sigmac[2, 2])
    # plt.plot(t_c, x_check[2], 'r', label='Corrected estimate $(\check x_k)_3$')
    # plt.fill_between(t_c, x_check[2] - sd_c, x_check[2] + sd_c, color='red', alpha=0.2)

    plt.xlabel('$k$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xlim([0, k_bar])
    plt.ylim([-x_max, x_max])

    plt.tight_layout()
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
Rv = 1                           # Process noise covariance (v_k ~ N(0, 1))
Rw = 1                            # Measurement noise covariance (w_k ~ N(0, 4))

# Compute LQR feedback gains
K, P = lqr_control(A, B_u, Q, R, Qf, k_bar)

# Initial state
Sigma = 25*np.eye(3)  # initial covariance
x0 = np.random.multivariate_normal(np.zeros(3), Sigma)  # Initial state x0 ~ N(0, I)

# Simulate LQR (no Σ needed) and LQG controllers
x_LQR, u_LQR, _, _, _, _, _  = simulate_lq_control(A, B_u, B_v, C, Sigma, K, k_bar, x0, mode='lqr',  Rw=Rw, Rv=Rv)
x_true, x_hat_LQG, u, y, Sigmas, x_check, Sigmac = simulate_lq_control(A, B_u, B_v, C, Sigma, K, k_bar, x0, mode='lqg_pred', Rw=Rw, Rv=Rv)

if __name__ == '__main__':
    # Plot the results with ±1σ shading
    figure5_3a(x_LQR, x_true, Sigmas, k_bar)
    figure5_3b(x_true, x_hat_LQG, y, Sigmas, x_check, Sigmac, k_bar)
    figure5_3c(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar)
    figure5_3d(x_true, x_hat_LQG, Sigmas, x_check, Sigmac, k_bar)