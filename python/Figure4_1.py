# Author: Kenji Kashima
# Date  : 2025/04/01
# Note  : pip install control

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
import sys
sys.path.append("./")
import config

a = 1.1;        # system matrix
alpha = -0.2;   # parameter in control law
x_max = 5;      # view range

def figure4_1a(n_k:int=50):
    '''
        n_k - total steps
    '''
    figsize = config.global_config(type=1)
    
    x = np.ones(n_k+1)       # state with v=0
    for k in range(n_k):
        u = alpha * (a + alpha) ** k  # control law
        x[k+1] = a * x[k] + u
    
    plt.figure(figsize=figsize)
    plt.plot(x,linewidth=1,color=[0,0,0])
    plt.xlim([0,n_k])
    plt.ylim([-x_max,x_max])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.tight_layout()
    plt.grid()
    plt.savefig("./figures/Figure4_1a.pdf")
    plt.show()

def figure4_1b(n_k:int=50,n_sample:int=20):
    '''
        n_k - total steps

        n_sample - number of samples
    '''
    figsize = config.global_config(type=1)

    x = np.ones(n_k+1)       # state with v=0
    x_v = np.ones([n_sample,n_k+1]) # state with v~N(0,1)
    for k in range(n_k):
        v = np.random.randn(n_sample) # n_sample independent standard normal distributions
        u = alpha * (a + alpha) ** k  # control law
        x[k+1] = a * x[k] + u
        x_v[:,k+1] = a * x_v[:,k] + u + 0.1 * v
    
    plt.figure(figsize=figsize)
    for i in range(n_sample):
        plt.plot(x_v[i,:],linewidth=0.2,color=[0.7,0.7,0.7])

    plt.plot(x,linewidth=1,color=[0,0,0])
    plt.xlim([0,n_k])
    plt.ylim([-x_max,x_max])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.tight_layout()
    plt.grid()
    plt.savefig("./figures/Figure4_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure4_1a(n_k=50)
    figure4_1b(n_k=50,n_sample=20)
