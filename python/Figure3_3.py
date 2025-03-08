# Author: Kenji Kashima
# Date  : 2023/02/24

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
import sys
sys.path.append("./")
import config

x_max = 10    # range of y_lim

def figure3_3a(n_k:int=50,n_sample:int=20,a:float=0.5):
    '''
        n_k - total steps

        n_sample - number of samples

        a - gain
    '''
    config.global_config()
    

    x= np.zeros([n_sample,n_k+1])
    x[:,0] = np.random.randn(n_sample).T

    step = np.random.randn(n_sample).T
    for t in range(n_k):
        
        x[:,t+1] =  a*x[:,t] + step
    
    plt.figure(figsize=(8,7))
    for x_sample in x:
        plt.plot(np.arange(0,n_k+1),x_sample, linewidth=2,alpha=0.8)

    plt.xlabel(r'$k$',)
    plt.ylabel(r'$x_k$')
    plt.ylim([-x_max,x_max])
    plt.xlim([0,n_k])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure3_3a.pdf")
    plt.show() 


def figure3_3b(n_k:int=50,n_sample:int=20,a:float=0.5):
    '''
        n_k - total steps

        n_sample - number of samples

        a - gain
    '''
    figsize = config.global_config(type=1)

    x= np.zeros([n_sample,n_k+1])
    x[:,0] = np.random.randn(n_sample).T

    for t in range(n_k):
        noise = np.random.randn(n_sample).T
        x[:,t+1] =  a*x[:,t] + noise*np.sqrt(3)
    
    plt.figure(figsize=figsize)
    for x_sample in x:
        plt.plot(np.arange(0,n_k+1),x_sample, linewidth=2,alpha=0.8)

    plt.xlabel(r'$k$')
    plt.ylabel(r'$x_k$')
    plt.ylim([-x_max,x_max])
    plt.xlim([0,n_k])
    plt.grid()    
    plt.tight_layout()
    plt.savefig("./figures/Figure3_3b.pdf")
    plt.show() 

if __name__ == '__main__':
    figure3_3a(n_k=10,n_sample=20)
    figure3_3b(n_k=10,n_sample=20)