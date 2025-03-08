# Author: Kenji Kashima
# Date  : 2023/03/12
# Note  : Used in Figure5_3

import numpy as np

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
