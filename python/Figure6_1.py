# Author: Kenji Kashima
# Date  : 2025/04/01
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov
import control

import sys
sys.path.append("./")
import config

def setup_system_matrices():
    """Set up system matrices A and B."""
    # System matrices
    A = np.array([[0.8, 0.9, 0.86],
                  [0.3, 0.25, 1],
                  [0.1, 0.55, 0.5]])  # A-matrix
    B = np.array([[1], [0], [0]])  # Input matrix B
    return A, B


def initialize_cost_matrices(n_x, n_u):
    """Initialize the cost matrices Q, R, and S."""
    QRS = np.eye(n_x + n_u)
    QRS = QRS @ QRS.T  # Ensure QRS is positive definite
    Q = QRS[:n_x, :n_x]  # Stage cost for states
    R = QRS[n_x:, n_x:]  # Stage cost for control inputs
    S = QRS[:n_x, n_x:]  # Cross term between state and control input
    return Q, R, S


def optimal_gain(A, B, Q, R, S, beta):
    """Compute the optimal gain using the discrete-time LQR approach."""
    _, P_opt, _ = control.dlqr(A * np.sqrt(beta), B * np.sqrt(beta), Q, R, S)
    return P_opt

def value_iteration(A, B, Q, R, S, P_opt, beta, N_iter):
    n_x = np.shape(A)[0]
    # Initialize error list for Value Iteration
    err_list_VI = np.zeros(N_iter)
    PI = np.zeros((n_x, n_x))
    # Value iteration
    i = 1
    while True:
        Rt = R + beta * B.T @ PI @ B
        St = S + beta * A.T @ PI @ B
        Qt = Q + beta * A.T @ PI @ A
        K = np.linalg.solve(Rt, St.T)  # Calculate K(Pi)
        if np.sqrt(beta) * np.max(np.abs(np.linalg.eigvals(A - B @ K ))) < 1:
            print(f"Stablilized after {i} iterations")
            break
        i += 1
        PI = Qt - St @ np.linalg.solve(Rt, St.T)  # Pi <- Ric(Pi)

    PI_ini = PI
    K_ini = K

    # Error list for further iterations
    PI = PI_ini
    for i in range(N_iter):
        err_list_VI[i] = np.linalg.norm(P_opt - PI)
        Rt = R + beta * B.T @ PI @ B
        St = S + beta * A.T @ PI @ B
        Qt = Q + beta * A.T @ PI @ A
        PI = Qt - St @ np.linalg.solve(Rt, St.T)    # PI <- Ric(Pi)

    return PI_ini, K_ini, err_list_VI

def policy_iteration(A, B, Q, R, S, P_opt, beta, K_ini, PI_ini, N_iter):
    # Policy iteration
    n_x = np.shape(A)[0]
    err_list_PI = np.zeros(N_iter)
    K = K_ini
    PI = PI_ini
    for i in range(N_iter):
        err_list_PI[i] = np.linalg.norm(P_opt - PI)
        PI_Q = solve_discrete_lyapunov(np.sqrt(beta) * (A - B * K).T,
                                    np.block([np.eye(n_x), - K.T]) @ np.block([[Q, S],[S.T, R]]) @ np.block([[np.eye(n_x)],[-K]]))
        Rt = R + beta * B.T @ PI_Q @ B
        St = S + beta * A.T @ PI_Q @ B
        Qt = Q + beta * A.T @ PI_Q @ A
        K = np.linalg.solve(Rt, St.T)   # - Upsilon_22^{-1} Upsilon_12'
        PI = Qt - St @ K  # Calculate V^pi from Q^pi for error analysis
    return err_list_PI
    
def Figure6_1(beta=0.95, N_iter=11):
    #beta = 0.95  # Discount rate for LQR
    n_x = 3  # Number of state variables
    n_u = 1  # Number of control iN_iter = 11  # Number of iterations

    
    # Set up system matrices
    A, B = setup_system_matrices()
    # Initialize cost matrices
    Q, R, S = initialize_cost_matrices(n_x, n_u)
    # Compute the optimal gain
    P_opt = optimal_gain(A, B, Q, R, S, beta)
    
    # Perform value iteration
    PI_ini, K_ini, err_list_VI = value_iteration(A, B, Q, R, S, P_opt, beta, N_iter)
    # Perform policy iteration
    err_list_PI = policy_iteration(A, B, Q, R, S, P_opt, beta, K_ini, PI_ini, N_iter)

    # Plotting results
    figsize = config.global_config(type=0)
    plt.figure(figsize=figsize)
   
    plt.grid()
    plt.plot(range(N_iter), err_list_VI, 'b', label='Value Iteration', linewidth=1.5)
    plt.plot(range(N_iter), err_list_PI, 'r', label='Policy Iteration', linewidth=1.5)
    plt.legend()
    plt.ylim([1e-9, 1000])
    plt.xlim([0, 10])
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.savefig("./figures/Figure6_1.pdf")
    plt.show()

if __name__ == "__main__":
    Figure6_1()