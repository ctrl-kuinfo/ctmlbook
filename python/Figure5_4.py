import numpy as np
import matplotlib.pyplot as plt
from cvxpy import Variable, Minimize, quad_form, Problem
from scipy.stats import truncnorm, laplace

np.random.seed(1)
import sys
sys.path.append("./")
import config

# System matrices and parameters setup
def initialize_matrices():
    k_bar = 60  # total steps

    # System matrices
    A = np.array([[-0.39, -0.67, -0.34],
                  [0.71, -0.51, 0.11],
                  [-0.46, -0.35, -0.12]])  # System matrix A
    B_v = np.array([[0], [1], [0]])  # Input matrix B_v
    C = np.array([[1, 0, 0]])  # Output matrix C

    # Initial state mean
    mu_0 = np.array([1, 1, 1])  # Mean of the initial state
    sigma_0 = np.eye(3)  # Covariance of the initial state

    return A, B_v, C, mu_0, sigma_0, k_bar

# Generate system dynamics and noise
def generate_data(A, B_v, C, mu_0, sigma_0, k_bar):
    # Noise parameters
    v_bounds = [-1, 1]  # Range for process noise v_k (truncated Gaussian)
    laplace_scale = 1  # Scale parameter for Laplace noise w_k

    # Generate the data sequence for x and y
    x = np.random.multivariate_normal(mu_0, sigma_0)  # Initial state x_0 ~ N(mu_0, I)
    x_data = np.zeros((3, k_bar+1))  # Store x_{0:k_bar}
    y_data = np.zeros(k_bar+1)  # Store y_{0:k_bar}

    x_data[:, 0] = x

    # Process noise and measurement noise
    v_list = []
    w_list = []

    for k in range(k_bar):
        # Generate truncated standard Gaussian noise v_k bounded in [-1, 1]
        v_k = truncnorm.rvs(v_bounds[0], v_bounds[1], loc=0, scale=1)

        # Generate Laplace distributed noise w_k ~ Lap(0, 1)
        w_k = laplace.rvs(scale=laplace_scale)

        # Update the system dynamics
        y_data[k] = C @ x + w_k  # Output measurement y_k
        x = A @ x.reshape(3, 1) + B_v * v_k  # State update x_{k+1}

        # Store the values for future use
        x_data[:, k+1] = x.flatten()
        v_list.append(v_k)
        w_list.append(w_k)

    # The last measurement noise w_{k_bar}
    w_k = laplace.rvs(scale=laplace_scale)
    w_list.append(w_k)
    y_data[k_bar] = C @ x + w_k

    return x_data, y_data, v_list, w_list

# Solve the optimization problem using cvxpy
def solve_optimization(A, B_v, C, mu_0, k_bar, y_data):
    n_l1 = k_bar + 1

    # Define optimization variables
    x0 = Variable(3)
    u = Variable(k_bar)
    t = Variable(n_l1)

    # Quadratic objective
    Q_x0 = np.eye(3)
    Q_u = np.eye(k_bar)

    # Objective: 1/2 * ||x0 - mu0||^2 + sum t_i
    objective = Minimize(quad_form(x0 - mu_0, Q_x0)/2 + quad_form(u,Q_u)/2 + sum(t))

    # Constraints: L1 norm and control input bounds
    constraints = []

    # L1 norm constraints
    Aux1 = np.zeros((k_bar + 1, 3))
    Aux2 = np.zeros((k_bar + 1, k_bar))

    for i in range(1, k_bar + 1):
        Aux1[i-1, :] = C @ np.linalg.matrix_power(A, i-1)
        for j in range(i-1):
            Aux2[i-1, j] = C @ np.linalg.matrix_power(A, i-j-2) @ B_v

    for i in range(n_l1):
        constraints += [Aux1[i, :] @ x0 + Aux2[i, :] @ u <= y_data[i] + t[i],
                        -Aux1[i, :] @ x0 - Aux2[i, :] @ u <= -y_data[i] + t[i]]

    # Control input bounds |u_k| <= 1
    for k in range(k_bar):
        constraints += [u[k] <= 1, u[k] >= -1]

    # Solve problem
    prob = Problem(objective, constraints)
    prob.solve(verbose=True)

    return x0.value, u.value, t.value, Aux1, Aux2

# Plot figures
def plot_figures(x_data, y_data, v_list, w_list, Aux1, Aux2, x0_est, u_est, k_bar):
    x_max = 6
    def figure5_4a():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.plot(range(k_bar+1), x_data[0, :], label=r'True state $(x_k)_1$', linewidth=1.5)
        plt.plot(range(k_bar+1), y_data, 'g--', label=r'Measurements $y$', linewidth=1.5)
        plt.plot(range(k_bar+1), Aux1 @ x0_est + Aux2 @ u_est, 'r-.', label=r'Estimated $(\hat{x}_k)_1$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./figures/Figure5_4a.pdf")
        plt.show()

    def figure5_4b():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.plot(range(k_bar+1), x_data[2, :], label=r'True state $(x_k)_3$', linewidth=1.5)
        plt.plot(range(k_bar+1), Aux1 @ x0_est + Aux2 @ u_est, 'r-.', label=r'Estimated $(\hat{x}_k)_3$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./figures/Figure5_4b.pdf")
        plt.show()

    def figure5_4c():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.step(range(k_bar), v_list, label=r'Disturbance $v_k$', linewidth=1.5)
        plt.step(range(k_bar), u_est, 'r-.', label=r'Estimated $\hat{v}_k:=u_k$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar-1])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./figures/Figure5_4c.pdf")
        plt.show()

    def figure5_4d():
        figsize = config.global_config(type=0)
        plt.figure(figsize=figsize)
        plt.step(range(k_bar+1), w_list, label=r'Noise $w_k$', linewidth=1.5)
        plt.step(range(k_bar+1), y_data - Aux1 @ x0_est - Aux2 @ u_est, 'r-.', label=r'Estimated $\hat{w}_k$', linewidth=1.5)
        plt.xlabel(r'$k$')
        plt.legend()
        plt.grid(True)
        plt.xlim([0,k_bar])
        plt.ylim([-x_max,x_max])
        plt.tight_layout()
        plt.savefig("./figures/Figure5_4d.pdf")
        plt.show()

    figure5_4a()
    figure5_4b()
    figure5_4c()
    figure5_4d()

# Main function to run the simulation and plotting
def main():
    A, B_v, C, mu_0, sigma_0, k_bar = initialize_matrices()
    x_data, y_data, v_list, w_list = generate_data(A, B_v, C, mu_0, sigma_0, k_bar)
    x0_est, u_est, t_est, Aux1, Aux2 = solve_optimization(A, B_v, C, mu_0, k_bar, y_data)
    plot_figures(x_data, y_data, v_list, w_list, Aux1, Aux2, x0_est, u_est, k_bar)

if __name__ == '__main__':
    main()
