# Author: Kenji Kashima
# Date  : 2023/03/12
# Note  : pip install cvxpy
#  (important!!!) You need to install cvxpy first!!!

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import Minimize,Variable,Problem
from cvxpy.atoms import quad_form,norm
from matplotlib.ticker import MultipleLocator

np.random.seed(1)
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


def figure7_4(n_data:int=30, n_x:int=100):
    '''
        n_sample - number of data

        n_x - divided x \in [0,1] by n_x parts
    '''
    figsize = config.global_config(type= 1)

    # generate data from U(0,1)
    x_s = np.random.rand(n_data)
    X = np.zeros([10,n_data])
    y_s=np.zeros(n_data)
    for i in range(n_data):
        X[:,i] = phi(x_s[i])
        y_s[i] = phi_real(x_s[i])+np.random.randn()
    X = X.T

    # hyper-parameter
    beta = 0.01

    # optimization 1 Naive

    theta = Variable(10)
    obj = Minimize(norm(X @ theta - y_s))
    prob = Problem(obj)
    prob.solve()
    theta_naive=theta.value

    # optimization 2 Lasso
    theta = Variable(10)
    obj = Minimize(norm(X @ theta - y_s) ** 2 /n_data+ beta * norm(theta,1) )
    prob = Problem(obj)
    prob.solve()
    theta_lasso=theta.value

    # optimization 3 Ridge
    theta = Variable(10)
    obj = Minimize(norm(X @ theta - y_s) ** 2/n_data + beta * norm(theta) ** 2)
    prob = Problem(obj)
    prob.solve()
    theta_ridge=theta.value


    x = np.linspace(0,1,n_x)
    NAIVE_dat = np.zeros(n_x)
    LASSO_dat = np.zeros(n_x) 
    RIDGE_dat = np.zeros(n_x)
    f_real    = np.zeros(n_x)
    for i in range(n_x):
        NAIVE_dat[i] = theta_naive.T @ phi(x[i])
        LASSO_dat[i] = theta_lasso.T @ phi(x[i])
        RIDGE_dat[i] = theta_ridge.T @ phi(x[i])
        f_real[i] =  phi_real(x[i])

    # Figure 7.4(a)
    plt.figure(figsize=figsize)
    plt.scatter(x_s,y_s,marker='o',label=r'${\rm y}_s$')
    plt.plot(x,NAIVE_dat,'k',linewidth=2,label='Least Square')
    plt.plot(x,RIDGE_dat,'b',linewidth=2,label='Ridge')
    plt.plot(x,LASSO_dat,'r',linewidth=2,label='Lasso')
    plt.plot(x,f_real,"-.",linewidth=2,label=r'$2\sin(5{\rm x}_s)$')
    plt.axis([0,1,-5,5])
    plt.xlabel(r'$\rm x$')
    plt.ylabel(r'$f({\rm x})$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure7_4a.pdf")
    
    
    # Figure 7.4(b)
    plt.figure(figsize=figsize)
    plt.scatter(np.arange(1,11),np.abs(theta_ridge),marker='x',s=60,clip_on=False,label='Ridge')
    plt.scatter(np.arange(1,11),np.abs(theta_lasso),marker='o',s=60,clip_on=False,label='Lasso')
    plt.xlabel(r'$i$')
    # plt.ylabel(r'$i$-th coefficient')
    plt.legend()
    plt.xlim([1,10])
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.grid()
    plt.savefig("./figures/Figure7_4b.pdf")

    plt.show()

if __name__ == '__main__':
    figure7_4(n_data=30, n_x=100)