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
    S = QRS[n_x:, :n_x]  # Cross term between state and control input
    return Q, R, S


def optimal_gain(A, B, Q, R, S, beta):
    """Compute the optimal gain using the discrete-time LQR approach."""
    _, P_opt, _ = control.dlqr(A * np.sqrt(beta), B * np.sqrt(beta), Q, R, S.T)
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
        St = S + beta * B.T @ PI @ A
        Qt = Q + beta * A.T @ PI @ A
        PIt = Qt - St.T @ np.linalg.solve(Rt, St)  # Solve for PIt
        Kt = np.linalg.solve(Rt, St)  # Calculate Kt
        if np.sqrt(beta) * np.max(np.abs(np.linalg.eigvals(A - B @ Kt))) < 1:
            print(f"Converged after {i} iterations")
            break
        i += 1
        PI = PIt

    PI_ini = PIt
    K_ini = Kt



    # Error list for further iterations
    PIt = PI_ini
    for i in range(N_iter):
        err_list_VI[i] = np.linalg.norm(P_opt - PIt)
        Rt = R + beta * B.T @ PI @ B
        St = S + beta * B.T @ PI @ A
        Qt = Q + beta * A.T @ PI @ A
        PIt = Qt - St.T @ np.linalg.solve(Rt, St)
        PI = PIt

    return PI_ini, K_ini, err_list_VI

def policy_iteration(A, B, Q, R, S, P_opt, beta, K_ini, PI_ini, N_iter):
    # Policy iteration
    n_x = np.shape(A)[0]
    err_list_PI = np.zeros(N_iter)
    Kt = K_ini
    PIt = PI_ini
    for i in range(N_iter):
        err_list_PI[i] = np.linalg.norm(P_opt - PIt)
        PI_Q = solve_discrete_lyapunov(np.sqrt(beta) * (A - B * Kt).T,
                                    np.block([np.eye(n_x), - Kt.T]) @ np.block([[Q, S.T],[S, R]]) @ np.block([[np.eye(n_x)],[-Kt]]))
        Rt = R + beta * B.T @ PI_Q @ B
        St = S + beta * B.T @ PI_Q @ A
        Qt = Q + beta * A.T @ PI_Q @ A
        Kt = np.linalg.solve(Rt, St)
        PIt = Qt - St.T @ Kt  # For Riccati error calculation
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