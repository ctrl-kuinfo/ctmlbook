# Author: Kenji Kashima
# Date  : 2025/05/25

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(13)

def gaussian_kernel(x,y,c):
    """
    Gaussian kernel function (RBF kernel) in eq 12.7 and 12.8. 

    Parameters:
        x : ndarray
            Input array (e.g., column vector of sample points).
        y : ndarray
            Input array to compare with x (will be transposed inside).
        c : float
            Kernel width (scale parameter).

    Returns:
        ndarray
            Kernel matrix K where K[i,j] = exp( - (x_i - y_j)^2 / c^2 )
    """
    return np.exp(-(x-y.T)**2/c**2)

def min_kernel(x,y):
    """
    Min kernel function in eq. 12.30 and 12.8.

    Parameters:
        x : ndarray
            Input array (1D, will be flattened).
        y : ndarray
            Input array (1D, will be flattened).
    
    Returns:
        ndarray
            Kernel matrix K where K[i,j] = min(x_i, y_j).
    """
    x = np.ravel(x)
    y = np.ravel(y)
    return np.minimum.outer(y, x)

def figure12_2(N_x= 100, c = 0.1,label="a"):
    '''  
        Figure12.2(a) c=0.1 
        Figure12.2(b) c=1.0
    '''
    figsize = config.global_config(type= 1)
    x = np.linspace(0,1,N_x).reshape(-1,1) # x for plot 
    # gaussian kernel function
    K = gaussian_kernel(x,x,c)     

    x = x.flatten()

    # prior distribution
    prior_mean = x # y=mu(x)=x

    plt.figure(figsize=figsize)

    for i in range(5): # 5 samples
         plt.plot(x,np.random.multivariate_normal(prior_mean,K),color="gray",alpha=0.8)  #eq 12.13
    plt.plot(x,prior_mean,label=r"$\mu({\rm x})$")  
    x_SD = np.sqrt(np.diag(K))   # standard deviation of x

    plt.fill_between(x,prior_mean-x_SD,prior_mean+x_SD,color="blue",alpha=0.25)
    plt.xlim([0,1])
    plt.ylim([-3,3])
    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure12_2{}.pdf".format(label))
    plt.show()

def figure12_3(N_x= 100, s_bar = 10, c = 0.1, label="a"):
    '''  
        Figure12.3(a) s_bar=10 
        Figure12.3(b) s_bar=50
    '''
    figsize = config.global_config(type= 1)
    x = np.expand_dims(np.linspace(0,1,N_x),axis=0)
    x_sample = np.random.rand(s_bar) * np.random.rand(s_bar); # x data points
    x_sample = np.expand_dims(x_sample,axis=0)
    
    # gaussian kernel function
    K = gaussian_kernel(x_sample, x_sample, c)  # eq 12.8
    kx = gaussian_kernel(x_sample, x, c)    # eq 12.9

    x = x.flatten()
    x_sample = x_sample.flatten()

    # function in example 12.1.9 and Fig 12.4
    mu_x = lambda x : x                     # mu(x)=x
    fn_x = lambda x :np.sin(4*np.pi*x)    # y(x) = sin(4πx)

    fn_list = fn_x(x)

    sigma = 0.1
    e_s = sigma * np.random.randn(s_bar)    # noise sample
    y_sample = fn_x(x_sample) + e_s         # observation data 

    plt.figure(figsize=figsize)
    plt.plot(x,mu_x(x),color = 'red' ,label=r"$\mu({\rm x})$") 
    plt.plot(x,fn_list,'k--',label=r'$\sin(4\pi {\rm x})$');  

    #plot samples
    plt.scatter(x_sample,y_sample,edgecolor='b',marker='o',facecolor='none', label=r"${\rm y}_s$")

    prior_mean = mu_x(x_sample)
    mean = mu_x(x)+kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ (y_sample-prior_mean) # eq 12.16
    Kpost = gaussian_kernel(x, x, c) - kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ kx.T  # eq 12.17
    x_SD = np.sqrt(np.diag(Kpost))  # posterior standard deviation

    plt.fill_between(x,mean-x_SD,mean+x_SD,color="blue",alpha=0.25)
    plt.plot(x,mean,"-.", color="blue",   label = r"$\mu ({\rm x}|\mathcal D)$")
    plt.xlim([0,1])
    plt.ylim([-1.5,3.0])
    plt.yticks([-1,0,1,2,3])

    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure12_3{}.pdf".format(label))
    plt.show()
    return (x_sample,y_sample)


def figure12_4(N_x= 100, s_bar = 10, label="a", sample0=None):
    '''  
        Figure12.4(a) s_bar=10 
        Figure12.4(b) s_bar=50
    '''
    figsize = config.global_config(type= 1)
    x=np.expand_dims(np.linspace(0,1,N_x),axis=0)

    # prior distribution
    mu_x = lambda x : x                      # mu(x)=x
    fn_x = lambda x : np.sin(4*np.pi*x)      # y(x) = sin(4πx)
    sigma = 0.1
    if sample0 is not None:
        x_sample,y_sample = sample0
        x_sample = np.expand_dims(x_sample,axis=0)
    else:
        x_sample= np.random.rand(s_bar) * np.random.rand(s_bar); # x data points
        x_sample = np.expand_dims(x_sample,axis=0)

        e_s = sigma * np.random.randn(s_bar)
        y_sample = fn_x(x_sample) + e_s
    
    # gaussian kernel function
    K = min_kernel(x_sample, x_sample) 
    kx = min_kernel(x_sample, x)

    x = x.flatten()
    x_sample = x_sample.flatten()
       
    fn_list = fn_x(x)
    plt.figure(figsize=figsize)
    plt.plot(x,mu_x(x),color = 'red' ,label=r"$\mu({\rm x})$") 
    plt.plot(x,fn_list,'k--',label=r'$\sin(4\pi {\rm x})$');  

    #plot samples
    plt.scatter(x_sample,y_sample,edgecolor='b',marker='o',facecolor='none', label=r"${\rm y}_s$")

    prior_mean = mu_x(x_sample)
    mean = mu_x(x) + kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ (y_sample-prior_mean) # eq 12.16
    Kpost = min_kernel(x, x) - kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ kx.T  # eq 12.17
    x_SD = np.sqrt(np.diag(Kpost))  # 

    plt.fill_between(x,mean-x_SD,mean+x_SD,color="blue",alpha=0.25)
    plt.plot(x,mean,"-.", color="blue",   label = r"$\mu ({\rm x}|\mathcal D)$")
    plt.xlim([0,1])
    plt.ylim([-1.5,3.0])
    plt.yticks([-1,0,1,2,3])
    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./Figure12_4{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    figure12_2(N_x=100, c = 0.1, label="a") #Figure12.2(a)
    figure12_2(N_x=100, c = 1.0, label="b") #Figure12.2(b)
    sample10 = figure12_3(N_x=100, s_bar = 10, label="a") #Figure12.3(a)
    sample50 = figure12_3(N_x=100, s_bar = 50, label="b") #Figure12.3(b)
    figure12_4(N_x=100, s_bar = 10, label="a", sample0 = sample10) #Figure12.4(a)
    figure12_4(N_x=100, s_bar = 50, label="b", sample0 = sample50) #Figure12.4(b)











