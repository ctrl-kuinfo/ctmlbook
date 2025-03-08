# Author: Kenji Kashima
# Date  : 2023/03/11
# Note  : pip install control

import control as ctl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
import sys
sys.path.append("./")
import config

def figure4_2(n_k:int = 800, T_c:float=0.01):
    '''
        n_k - total steps

        T_c - discrete-time stepsize [s]
    '''
    figsize = config.global_config(type=2)

    # parameters for a continuous-time system
    A = np.array([[0,4],[-3,2]])
    B = np.array([[0],[1]])
    C = np.array([-0.3,-4])
    csys = ctl.ss(A,B,C,0) # build CT system

    dsys = ctl.c2d(csys, T_c)         # build DT system from the CT system
    [Ad,Bd,Cd,Dd] = ctl.ssdata(dsys)  # get parameters of the DT system
                                      # Note: Dd=0
    n_k = 800      # total steps
    u_max = 3.2    # view range
    d = 1.0        # quantization unit
    x0 = np.array([[1],[0.5]])  # init state

    fig = plt.figure(figsize=figsize)
    # u_k = y_k
    y = np.zeros(n_k)
    x_k = x0
    for k in range(n_k):
        y[k] = float(Cd @ x_k)
        u_k = y[k]
        x_k = Ad @ x_k + Bd * u_k

    ax = fig.subplots(1,3)
    ax[0].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[0].set_title(r'$u_k=y_k$')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel(r'$k$')
    ax[0].tick_params()
    ax[0].axis([0,n_k,-u_max,u_max])

    # u_k = Q(y_k)
    y = np.zeros(n_k)
    u = np.zeros(n_k)
    x_k = x0
    for k in range(n_k):
        y[k] = float(Cd @ x_k)
        u[k] = np.floor((y[k] + d/2) / d) * d
        x_k = Ad @ x_k + Bd * u[k]
    ax[1].stairs(u,label=r'$u_k$',linewidth=1.0, color = [0.7,0.7,0.7],zorder=10)
    ax[1].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[1].set_title(r'$u_k=\mathcal{Q}(y_k)$')
    ax[1].legend()
    ax[1].grid(zorder=0)
    ax[1].set_xlabel(r'$k$')
    ax[1].axis([0,n_k,-u_max,u_max])
    
    # u_k = Q(y_k + z_k)
    y = np.zeros(n_k)
    u = np.zeros(n_k)
    x_k = x0
    for k in range(n_k):
        y[k] = float(Cd @ x_k)
        z_k = np.random.rand() - 0.5 # uniform distribution [-0.5,0.5]
        u[k] = np.floor((y[k] + z_k + d/2) / d) * d
        x_k = Ad @ x_k + Bd * u[k]
    ax[2].stairs(u,label=r'$u_k$',linewidth=1.0, color = [0.7,0.7,0.7],zorder=10)
    ax[2].stairs(y,label=r'$y_k$',linewidth=1.0, color = 'black',zorder=10)

    ax[2].set_title(r'$u_k=\mathcal{Q}(y_k+z_k)$')
    ax[2].legend()
    ax[2].grid(zorder=0)
    ax[2].set_xlabel(r'$k$')
    ax[2].tick_params()
    ax[2].axis([0,n_k,-u_max,u_max])

    plt.tight_layout()
    plt.savefig("./figures/Figure4_2.pdf")
    plt.show()
    
if __name__ == '__main__':
    figure4_2()