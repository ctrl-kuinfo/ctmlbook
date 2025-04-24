# Author: Kenji Kashima
# Date  : 2023/03/12

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)
import sys
sys.path.append("./")
import config

def phi(x:float)->np.ndarray:
    '''
        x - input
    '''
    return  np.array([1, x, x**2, x**3, x**4, 
                      x**5, x**6, x**7, x**8, x**9])


def phi_real(x:float)->float:
    '''
        x - input

    '''
    return 2*np.sin(5*x)

def figure7_1a(n_x:int = 100, n_sample:int=20):
    '''
        n_x - divided x \in [0,1] by n_x parts

        n_sample - number of samples
    '''
    figsize = config.global_config(type=1)

    x = np.linspace(0,1,n_x)
    X = np.zeros([10,n_x]) 
    for i in range(n_x):
        X[:,i] = phi(x[i]) 

    plt.figure(figsize=figsize)
    plt.plot(x, (np.zeros([1,10]) @ X).flatten(), linewidth= 3)

    for i in range (n_sample):
        theta = np.random.randn(10)      
        fx =  theta @ X                 
        plt.plot(x,fx,linewidth= 0.5, color=[0.7,0.7,0.7])    
    
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.xlim([0,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure7_1a.pdf")
    plt.show()

# learning from n_data in one sample trajectory with hyperparameter sigma
def learn_theta(sigma,n_data):
    '''
        sigma - hyper-parameter ;n_sample - number of data
    '''
    x_s = np.linspace(0,1,n_data)
    X = np.zeros([10,n_data])
    y_s=np.zeros(n_data)
    for i in range(n_data):
        X[:,i] = phi(x_s[i])
        y_s[i] = phi_real(x_s[i]) + np.random.randn()

    size_sigma = len(sigma)
    mean = np.zeros([10,size_sigma])
    cov = np.zeros([10,size_sigma*10])

    for i in range(size_sigma):
        tmp = np.eye(10) - X @ np.linalg.inv(X.T @ X+sigma[i] * np.eye(n_data)) @ X.T
        cov[:,10*i:10*(i+1)] = (tmp+tmp.T)/2
        mean[:,i]  = X @ np.linalg.inv(X.T @ X + sigma[i] * np.eye(n_data)) @ y_s.T
    return mean, cov, x_s, y_s

def figure7_1b(n_x:int = 100, n_sample:int=20, n_data:int=8):
    '''
        n_x - divided x \in [0,1] by n_x parts

        n_sample - number of samples

        n_sample - number of data
    '''
    figsize = config.global_config(type=1)
    
    x = np.linspace(0,1,n_x)
    X = np.zeros([10,n_x]) 
    for i in range(n_x):
        X[:,i] = phi(x[i]) 
    sigma = [0.25,100,10**(-6)]  # hyper parameters
    mu,Sigma,x_s,y_s = learn_theta(sigma,n_data)

    plt.figure(figsize=figsize)

    plt.plot(x, (mu[:,0] @ X).flatten(), linewidth= 2,label=r'$\sigma=0.5$',zorder=10)
    plt.plot(x, (mu[:,1] @ X).flatten(), linewidth= 2,label=r'$\sigma=10$')
    plt.plot(x, (mu[:,2] @ X).flatten(), linewidth= 2,label=r'$\sigma=10^{-3}$')

    for i in range (n_sample):
        theta = np.random.multivariate_normal(mu[:,0],Sigma[:,0:10])      
        fx =  theta @ X                   
        plt.plot(x,fx,linewidth= 0.3,color=[0.7,0.7,0.7])

    plt.scatter(x_s,y_s,marker='o',label=r'${\rm y}_s$')
    plt.legend()
    
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.xlim([0,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure7_1b.pdf")
    plt.show()

if __name__ == '__main__':
    figure7_1a(n_x=100,n_sample=20)
    figure7_1b(n_x=100,n_sample=20,n_data=20)
