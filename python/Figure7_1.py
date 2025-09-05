# Author: Kenji Kashima
# Date  : 2025/04/01

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import config

np.random.seed(24)

def phi(x:float)->np.ndarray:
    '''
        Feature mapping
        x - input
    '''
    return  np.array([1, x, x**2, x**3, x**4, 
                      x**5, x**6, x**7, x**8, x**9 ])

n_f = phi(0).size



def f_true(x:float)->float:
    '''
        x - input
    '''
    return 2*np.sin(5*x)

def figure7_1a(n_x:int = 100, n_sample:int=20):
    '''
        n_x - number of x-grid for plot
        n_sample - number of sample functions
    '''
    figsize = config.global_config(type=1)

    x_p = np.linspace(0,1,n_x)
    X = np.zeros([n_f,n_x]) 
    for i in range(n_x):
        X[:,i] = phi(x_p[i]) 

    plt.figure(figsize=figsize)
    plt.plot(x_p, np.zeros_like(x_p), linewidth=3)

    for _ in range (n_sample):
        theta = np.random.randn(n_f)      
        fx =  theta @ X                 
        plt.plot(x_p,fx,linewidth= 0.5, color=[0.7,0.7,0.7])    
    
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.xlim([0,1])
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure7_1a.pdf")
    plt.show()

# learning from n_data in one sample trajectory with hyperparameter sigma_sq
def learn_theta(sigma_sq,n_data):
    '''
        sigma_sq - simulation for several weights (=standard deviation)^2
        n_data - number of data
    '''
    x = np.linspace(0,1,n_data)
    X = np.zeros([n_f,n_data])
    y=np.zeros(n_data)
    for s in range(n_data):
        X[:,s] = phi(x[s])
        y[s] = f_true(x[s]) + np.random.randn()

    size_sigma_sq = len(sigma_sq)
    mean = np.zeros([n_f,size_sigma_sq])
    cov = np.zeros([n_f,size_sigma_sq*n_f])

    for l in range(size_sigma_sq):
        tmp = np.eye(n_f) - X @ np.linalg.inv(X.T @ X+sigma_sq[l] * np.eye(n_data)) @ X.T
        cov[:,n_f*l:n_f*(l+1)] = (tmp+tmp.T)/2
        mean[:,l]  = X @ np.linalg.inv(X.T @ X + sigma_sq[l] * np.eye(n_data)) @ y.T
    return mean, cov, x, y

def figure7_1b(n_x:int = 100, n_sample:int=20, n_data:int=8):
    '''
        n_x - numer of x-grid for plot
        n_sample - number of sample functions
        n_data - number of data
    '''
    figsize = config.global_config(type=1)
    
    x_p = np.linspace(0,1,n_x)
    X = np.zeros([n_f,n_x]) 
    for i in range(n_x):
        X[:,i] = phi(x_p[i]) 
    sigma_sq = [0.25,100,10**(-6)]  # standard deviation sigma = 0.5, 10^3, 10^{-3}
    mu,Sigma,x,y = learn_theta(sigma_sq,n_data)

    plt.figure(figsize=figsize)

    plt.plot(x_p, (mu[:,0] @ X).flatten(), linewidth= 2,label=r'$\sigma=0.5$',zorder=10)
    plt.plot(x_p, (mu[:,1] @ X).flatten(), linewidth= 2,label=r'$\sigma=10$')
    plt.plot(x_p, (mu[:,2] @ X).flatten(), linewidth= 2,label=r'$\sigma=10^{-3}$')

    for _ in range (n_sample):
        theta = np.random.multivariate_normal(mu[:,0],Sigma[:,0:n_f])      
        fx =  theta @ X                   
        plt.plot(x_p,fx,linewidth= 0.3,color=[0.7,0.7,0.7])

    plt.scatter(x,y,marker='o',label=r'${\rm y}_s$')
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
