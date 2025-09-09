# Author: Kenji Kashima
# Date  : 2025/09/01
# Note  : pip install cvxpy

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_ellipse(Sigma):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    
    # Calculate rotation angle in radians
    rotation_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Calculate the lengths of the axes
    axes_lengths = 2 * np.sqrt(eigenvalues)

    # Generate theta values
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Calculate the x and y coordinates of the ellipse
    x = axes_lengths[0] * np.cos(theta) * np.cos(rotation_rad) - axes_lengths[1] * np.sin(theta) * np.sin(rotation_rad)
    y = axes_lengths[0] * np.cos(theta) * np.sin(rotation_rad) + axes_lengths[1] * np.sin(theta) * np.cos(rotation_rad)
    
    return x, y

def figure11_1(label="a"):
    figsize = config.global_config(type= 1)
    # Define parameters
    A = np.array([[1, 0.1], [-0.3, 1]])  # State matrix
    B = np.array([[0.7], [0.4]])         # Control matrix
    W = 0.1 * np.eye(2)                  # Noise covariance matrix
    Sigma_10 = np.array([[2, 0], [0, 0.5]])  # Target covariance
    n,m = B.shape
    N = 10  # Time horizon

    # Define optimization variables
    Sigma = [cp.Variable((n, n), PSD=True) for _ in range(N+1)]  # Sigma_k, k=0 to N
    P = [cp.Variable((n,m)) for _ in range(N)]  # P_k, k=0 to N-1
    M = [cp.Variable((m,m)) for _ in range(N)]  # M_k, k=0 to N-1, non-negative

    # Initial and terminal conditions
    constraints = [Sigma[0] == 3 * np.eye(n)]  # Sigma_0 = 3 * I
    constraints += [Sigma[N] == Sigma_10]  # Sigma_10
    if label!="a":
        constraints += [Sigma[5][1,1] <= 0.5]

    # Recurrence relation constraints
    for k in range(N):
        # Sigma_{k+1} = A * Sigma_k * A' + A * P_k * B' + B * P_k' * A' + B * M_k * B' + W
        Sigma_next = A @ Sigma[k] @ A.T + A @ P[k] @ B.T + B @ P[k].T @ A.T + B @ M[k] @ B.T + W
        constraints += [Sigma[k+1] == Sigma_next]
        
        # Semi-definite constraint: [Sigma_k, P_k; P_k', M_k] >= 0
        LMI = cp.bmat([[Sigma[k], P[k]], [P[k].T, M[k]]])
        constraints += [LMI >> 0]  # Ensure LMI is positive semi-definite

    # Objective function: minimize trace(sum(M_k))
    objective = cp.Minimize(cp.trace(cp.sum(M)))

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  # You can use SCS or other SDP solvers

    # Check the feasibility and output results
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        Sigma_optimal = [Sigma[k].value for k in range(N+1)]
        P_optimal = [P[k].value for k in range(N)]
        M_optimal = [M[k].value for k in range(N)]
        
        # Number of trajectories
        num_trajectories = 20
        trajectories = []

        for _ in range(num_trajectories):
            # Generate random initial state x_0 ~ N(0, 3I)
            x = np.random.multivariate_normal(mean=np.zeros(n), cov=3 * np.eye(n))
            trajectory = [x]
        
            for k in range(N):
                K_k = np.linalg.solve(Sigma_optimal[k], P_optimal[k]).T  # Compute optimal gain K_k
                v_k = np.random.multivariate_normal(mean=np.zeros(n), cov=W)  # Noise v_k ~ N(0, W)
                u_k = K_k @ x  # Control input based on the state
                x = A @ x + B @ u_k + v_k  # State update
                trajectory.append(x)
            
            trajectories.append(np.array(trajectory))

        # Plotting the trajectories in 3D
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=figsize)

        # Plot each trajectory
        for i in range(num_trajectories):
            ax.plot(list(range(N+1)), trajectories[i][:, 0], zs=trajectories[i][:, 1], color="gray",alpha=0.5)
        
        for i in range(N+1):
            x,y = plot_ellipse(Sigma_optimal[i])
            ax.plot(np.ones(100)*i, x, zs=y,alpha=0.9)
        ax.set_xlabel(r'$k$',labelpad=20)
        ax.set_ylabel(r'$(x)_1$',labelpad=20)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r'$(x)_2$',labelpad=10, rotation=90)

        ax.view_init(20, 70)
        ax.set_yticks(np.arange(-10, 11, 5))
        ax.set_zticks(np.arange(-10, 11, 5))
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_ylim([-10,10])  
        ax.set_zlim([-10,10])  
        ax.set_xlim([10,0])  
        plt.tight_layout()
        plt.savefig("./Figure11_1{}.pdf".format(label))
        plt.show()

if __name__ == '__main__':
    figure11_1(label="a")
    figure11_1(label="b")