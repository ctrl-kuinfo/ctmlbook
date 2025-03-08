# Author: Kenji Kashima
# Date  : 2023/09/30
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov as dlyap

# Random seed
np.random.seed(2)
import sys
sys.path.append("./")
import config

# Function for the RL4LQR algorithm
def RL4LQR(sigma, iter_Gain, iter_RLS, Npath):
    beta = 0.95  # discount rate (LQR for 1)
    A = np.array([[0.80, 0.90, 0.86],
                  [0.30, 0.25, 1.00],
                  [0.10, 0.55, 0.50]])  # A-matrix
    B = np.array([[1], [0], [0]])  # B-matrix

    x_dim, u_dim = B.shape  # state & input dimension
    n_theta = (x_dim + u_dim) * (x_dim + u_dim + 1) // 2  # size of theta

    E = np.eye(x_dim)  # cost x'Ex
    F = np.eye(u_dim)  # cost u'Fu

    # Optimal gain
    K_opt = -np.linalg.inv(np.dot(B.T, B) + F) @ B.T @ A
    K_ini = np.array([[-4.1100, -11.7519, -19.2184]])
    IsUnstable = False

    x_norm_hist_p = []
    Herr_hist_p = []
    Kerr_hist_p = []

    for j in range(Npath):
        
        Kerr_hist = []
        Herr_hist = []

        theta = np.zeros(n_theta)

        K = K_ini
        x = np.zeros((3, 1))  # Initial state of dynamics
        x_hist = x
        for k in range(iter_Gain):
            Pi = dlyap(np.sqrt(beta) * (A + B @ K).T, np.hstack([np.eye(x_dim), K.T]) @ np.vstack([np.eye(x_dim), K]))
            Qopt = beta * np.hstack([A, B]).T @ Pi @ np.hstack([A, B]) + np.block([[E, np.zeros((x_dim, u_dim))], [np.zeros((u_dim, x_dim)), F]])

            Knext = -np.linalg.inv(Qopt[x_dim:x_dim + u_dim, x_dim:x_dim + u_dim]) @ Qopt[x_dim:x_dim + u_dim, :x_dim]

            theta = np.zeros((n_theta,1))
            P = np.eye(n_theta) * 10  # RLS initialization
            for i in range(iter_RLS):
               

                Kerr_hist.append(np.linalg.norm(K - K_opt) / np.linalg.norm(K_opt))
                Herr_hist.append(np.linalg.norm(theta_to_H(theta, x_dim + u_dim) - Qopt) / np.linalg.norm(Qopt))

                u = K @ x + np.random.randn(u_dim, 1) * sigma  # tentative SF + exploration noise
                cost = x.T @ E @ x + u.T @ F @ u  # stage cost

                bar = phi(x, u)
                x = A @ x  + B @ u  # One-step simulation
                u = K @ x
                barplus = phi(x, u)
                x_hist= np.hstack((x_hist,x))

                phi_k = bar - beta * barplus
                e = cost - phi_k.T @ theta  # around (9.37)
                denom = 1 + phi_k.T @ P @ phi_k
                theta = theta + (P @ phi_k @ e) / denom
                P = P - (P @ phi_k @ phi_k.T @ P) / denom

            H = theta_to_H(theta, x_dim + u_dim)
            K = -np.linalg.inv(H[x_dim:x_dim + u_dim, x_dim:x_dim + u_dim]) @ H[x_dim:x_dim + u_dim, :x_dim]

            if max(abs(np.linalg.eigvals(A + np.dot(B, K)))) > 1:
                IsUnstable = True
                break

        if not IsUnstable:
            x_norm_hist_p.append(np.sqrt(np.diag(np.dot(np.array(x_hist).T, np.array(x_hist)))))
            Kerr_hist_p.append(Kerr_hist)
            Herr_hist_p.append(Herr_hist)
        else:
            IsUnstable = False
    return (x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain)
    

# Function to convert H matrix to theta vector
def phi(x, u):
    H = np.vstack([x, u])@ np.vstack([x, u]).T
    phi_vector = np.hstack([H[i, i:] for i in range(H.shape[0])])
    return phi_vector.reshape(-1,1)

# Function to convert theta vector to H matrix
def theta_to_H(theta, n):
    theta = theta.reshape(1,-1)
    H = np.zeros((n, n))
    idx = 0
    for i in range(n):
        H[i, i:] = theta[:,idx:idx + n - i]
        idx += n - i
    return (H + H.T) / 2

# Function to plot results
def plot_results(x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain, label=None):
    grayColor = [0.7, 0.7, 0.7]
    figsize = config.global_config(type= 1)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    
    # Plot x_norm_hist_p
    ax[0].plot(np.array(x_norm_hist_p).T, color=grayColor)
    ax[0].plot(np.mean(x_norm_hist_p, axis=0), linewidth=2, color='black')
    ax[0].set_xlim([0,len(np.array(x_norm_hist_p).T)-1])
    ax[0].axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Plot Herr_hist_p
    ax[1].plot(np.array(Herr_hist_p).T, color=grayColor)
    ax[1].plot(np.mean(Herr_hist_p, axis=0), linewidth=1.5, color='black')
    ax[1].set_ylabel('Q function error')
    ax[1].set_xlim([0,len(np.array(Herr_hist_p).T)-1])
    ax[1].axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(r'$k$')
    # Plot Kerr_hist_p if iter_Gain > 1
    if iter_Gain > 1:
        ax2 = ax[1].twinx()  # Create a twin y-axis
        ax2.plot(np.mean(Kerr_hist_p, axis=0), linewidth=2, color='red')
        ax2.set_ylabel('Gain error', color='red')

    plt.tight_layout()
    
    if label is not None:
        plt.savefig("./figures/Figure9_{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    # Running the RL4LQR for different scenarios
    # Figure 9.3(a)
    sigma = 2
    iter_Gain = 1
    iter_RLS = 50
    Npath = 20
    x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, iter_RLS, Npath)
    plot_results(x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain,label="3a")
    # Figure 9.3(b)
    sigma = 10
    x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, iter_RLS, Npath)
    plot_results(x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain,label="3b")

    # Figure 9.4
    iter_Gain = 5
    x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain = RL4LQR(sigma, iter_Gain, iter_RLS, Npath)
    plot_results(x_norm_hist_p, Kerr_hist_p, Herr_hist_p, iter_Gain, label="4")
