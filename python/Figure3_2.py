# Author: Kenji Kashima
# Date  : 2025/04/01

import numpy as np
import matplotlib.pyplot as plt
import sys
import config

np.random.seed(100)

def figure3_2a(n_k:int=50,n_sample:int=9):
    '''
        n_k - total steps
        n_sample - number of samples
    '''
    figsize = config.global_config(type=1)

    plt.figure(figsize=figsize)
    for i in range(n_sample):
        x = np.zeros(n_k)
        x[0] = -0.8+0.2*i # initialization
        for k in range(n_k-1):
             x[k+1] =  x[k] + 0.1*(x[k]-x[k]**3)
        plt.plot(x,linewidth=2)
    plt.xlim([1,n_k-1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.ylim([-1,1])
    plt.tight_layout()
    plt.grid()
    plt.savefig("./Figure3_2a.pdf")
    plt.show()

def figure3_2b(n_k:int=50,n_sample:int=10):
    '''
        n_k - total steps
        n_sample - number of samples
    '''
    figsize = config.global_config(type=1)

    plt.figure(figsize=figsize)
    for _ in range(n_sample):
        x = np.zeros(n_k)
        x[0] = -0.5 # initialization
        # Note: rand is a single uniformly distributed random 
        # number in the interval (0,1).
        for k in range(n_k-1):
             rand = np.random.rand()
             x[k+1] =  x[k] + 0.1*(x[k]-x[k]**3) + (rand-0.5)*(1-np.abs(x[k])) 
        plt.plot(x,linewidth=2)
    plt.xlim([0,n_k-1])
    plt.ylim([-1,1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.tight_layout()
    plt.grid()
    plt.savefig("./Figure3_2b.pdf")
    plt.show()

if __name__ == '__main__':
    figure3_2a(n_k=50,n_sample=9)
    figure3_2b(n_k=50,n_sample=10)
