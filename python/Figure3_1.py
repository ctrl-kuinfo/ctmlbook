# Author: Kenji Kashima
# Date  : 2025/02/11

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import config

np.random.seed(100)

def figure3_1a(n_k=9):
    '''
        n_k - number of k
    '''
    figsize = config.global_config(type=1)

    n_tmp = 1000
    n_x = 2*n_tmp+1 # number of x
    x = np.linspace(-1,1,n_x)
    P = np.zeros([n_x,n_x])
    for i in range(n_x):
        tmp_u = x[i] + 0.1*(x[i]-x[i]**3) + 0.5*(1-np.abs(x[i]))
        tmp_l = x[i] + 0.1*(x[i]-x[i]**3) - 0.5*(1-np.abs(x[i]))
        index_u = int(np.ceil((tmp_u+1)*n_tmp))+1
        index_l = int(np.floor((tmp_l+1)*n_tmp))+1    
        for j in range(index_l,index_u):
            P[j,i]=P[j,i]+1/(index_u-index_l+1)

    states = np.zeros([n_x,n_k+2]); 
    states[n_tmp+1,0]=1;  #initialization
    for i in range(n_k+1):
        states[:,i+1] = P @ states[:,i]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=figsize)
    for i in range(1,n_k+1):
        k=np.ones(n_x)*(i-1)
        phi=states[:,i]*n_x/2
        ax.plot3D(k,x.T,phi)
    ax.set_xlabel(r'$k$',labelpad=20)
    ax.set_ylabel(r'${\rm x}$',labelpad=20)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$\varphi_{x_k}$',rotation=90,labelpad=20)
    ax.view_init(26, -107)
    ax.set_yticks([-1,-0.5,0,0.5,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure3_1a.pdf")
    plt.show()

def figure3_1b(n_k:int=50,n_sample:int=10):
    '''
        n_k - total steps

        n_sample - number of samples
    '''
    config.global_config()

    plt.figure(figsize=(8,7))
    for _ in range(n_sample):
        x = np.zeros(n_k)
        rand = np.random.rand()
        x[0] = rand-0.5 # initialization
        # Note: rand is a single uniformly distributed random 
        # number in the interval (0,1).
        for k in range(n_k-1):
             rand = np.random.rand()
             x[k+1] =  x[k] + 0.1*(x[k]-x[k]**3) + (rand-0.5)*(1-np.abs(x[k])) 
        plt.plot(x,linewidth=2)
    plt.xlim([0,n_k-1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.grid()
    plt.ylim([-1,1])
    plt.tight_layout()
    plt.savefig("./figures/Figure3_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure3_1a(n_k=9)
    figure3_1b(n_k=30,n_sample=10)
