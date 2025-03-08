# Author: Kenji Kashima
# Date  : 2023/11/30
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)
import sys
sys.path.append("./")
import config

def gaussian_kernel(x,y,c):
    return np.exp(-(x-y.T)**2/c**2)

def min_kernel(x,y):
    return np.min((np.ones_like(y.T) @ x, y.T @ np.ones_like(x)),axis=0) # eq 13.30

def figure12_2(N_x= 100, c = 0.1,label="a"):
    '''  
        Figure12.2(a) c=0.1 
        Figure12.2(b) c=1.0
    '''
    figsize = config.global_config(type= 1)
    x_list=np.expand_dims(np.linspace(0,1,N_x),axis=0)
    # gaussian kernel function
    K =  gaussian_kernel(x_list,x_list,c) # eq 13.7 & 13.8 

    x_list = x_list.flatten()

    # original function in example 13.1.7
    y_list = x_list # y=mu(x)=x


    plt.figure(figsize=figsize)

    for i in range(5): # 5 samples
         plt.plot(x_list,np.random.multivariate_normal(y_list,K),color="gray",alpha=0.8)  #eq 13.13
    plt.plot(x_list,y_list,label=r"$\mu({\rm x})$")  
    kxx=np.sqrt(np.diag(K))
    
    plt.fill_between(x_list,y_list-kxx,y_list+kxx,color="blue",alpha=0.25)
    plt.xlim([0,1])
    plt.ylim([-3,3])
    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure12_2{}.pdf".format(label))
    plt.show()

def figure12_3(N_x= 100, s_bar = 10, c = 0.1, label="a"):
    '''  
        Figure12.3(a) s_bar=10 
        Figure12.3(b) s_bar=50
    '''
    figsize = config.global_config(type= 1)
    x_bar=np.expand_dims(np.linspace(0,1,N_x),axis=0)
    x_sample= np.random.rand(s_bar) * np.random.rand(s_bar); # take x samples randomly
    x_sample = np.expand_dims(x_sample,axis=0)

    # gaussian kernel function
    K = gaussian_kernel(x_sample, x_sample, c) # eq 13.7 & 13.8
    kx = gaussian_kernel(x_sample, x_bar, c)
    kxx = gaussian_kernel(x_bar, x_bar, c)

    x_bar = x_bar.flatten()
    x_sample = x_sample.flatten()

    # original function in example 13.1.7 and example 13.1.9
    mu_x = lambda x_bar : x_bar                     # mu(x)=x
    fn_x = lambda x_bar :np.sin(4*np.pi*x_bar)    # y(x) = sin(4πx)

    mu_list = mu_x(x_bar)
    fn_list = fn_x(x_bar)

    sigma = 0.1
    e_s = sigma * np.random.randn(s_bar)
    y_sample = fn_x(x_sample) + e_s

    plt.figure(figsize=figsize)
    plt.plot(x_bar,mu_list,color = 'red' ,label=r"$\mu({\rm x})$") 
    plt.plot(x_bar,fn_list,'k--',label=r'$\sin(4\pi {\rm x})$');  

    #plot samples
    plt.scatter(x_sample,y_sample,edgecolor='b',marker='o',facecolor='none', label=r"${\rm y}_s$")

    y_mean = mu_x(x_sample)
    mean= mu_x(x_bar)+kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ (y_sample-y_mean) # eq 13.16
    v=kxx-kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ kx.T  # eq 13.17
    vm=np.sqrt(np.diag(v))

    plt.fill_between(x_bar,mean-vm,mean+vm,color="blue",alpha=0.25)
    plt.plot(x_bar,mean,"-.", color="blue",   label = r"$\mu ({\rm x}|\mathcal D)$")
    plt.xlim([0,1])
    plt.ylim([-1.5,3.0])
    plt.yticks([-1,0,1,2,3])

    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure12_3{}.pdf".format(label))
    plt.show()
    return (x_sample,y_sample)


def figure12_4(N_x= 100, s_bar = 10, label="a", sample0=None):
    '''  
        Figure12.4(a) s_bar=10 
        Figure12.4(b) s_bar=50
    '''
    figsize = config.global_config(type= 1)
    x_bar=np.expand_dims(np.linspace(0,1,N_x),axis=0)

    # original function in example 13.1.7 and example 13.1.9
    mu_x = lambda x_bar : x_bar                     # mu(x)=x
    fn_x = lambda x_bar :np.sin(4*np.pi*x_bar)    # y(x) = -sin(4πx)
    sigma = 0.1
    if sample0 is not None:
        x_sample,y_sample = sample0
        x_sample = np.expand_dims(x_sample,axis=0)
    else:
        x_sample= np.random.rand(s_bar) * np.random.rand(s_bar); # take x samples randomly
        x_sample = np.expand_dims(x_sample,axis=0)

        
        e_s = sigma * np.random.randn(s_bar)
        y_sample = fn_x(x_sample) + e_s
    
    # gaussian kernel function
    K = min_kernel(x_sample, x_sample) 
    kx = min_kernel(x_sample, x_bar)
    kxx = min_kernel(x_bar, x_bar)

    x_bar = x_bar.flatten()
    x_sample = x_sample.flatten()
       
    mu_list = mu_x(x_bar)
    fn_list = fn_x(x_bar)
    plt.figure(figsize=figsize)
    plt.plot(x_bar,mu_list,color = 'red' ,label=r"$\mu({\rm x})$") 
    plt.plot(x_bar,fn_list,'k--',label=r'$\sin(4\pi {\rm x})$');  

    #plot samples
    plt.scatter(x_sample,y_sample,edgecolor='b',marker='o',facecolor='none', label=r"${\rm y}_s$")

    y_mean = mu_x(x_sample)
    mean= mu_x(x_bar)+kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ (y_sample-y_mean) # eq 13.16
    v=kxx-kx @ np.linalg.inv(K + sigma**2 * np.eye(s_bar)) @ kx.T  # eq 13.17
    vm=np.sqrt(np.diag(v))

    plt.fill_between(x_bar,mean-vm,mean+vm,color="blue",alpha=0.25)
    plt.plot(x_bar,mean,"-.", color="blue",   label = r"$\mu ({\rm x}|\mathcal D)$")
    plt.xlim([0,1])
    plt.ylim([-1.5,3.0])
    plt.yticks([-1,0,1,2,3])
    plt.xlabel(r"${\rm x}$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure12_4{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    figure12_2(N_x=100,c = 0.1, label="a") #Figure12.2(a)
    figure12_2(N_x=100,c = 1.0, label="b") #Figure12.2(b)
    sample10 =figure12_3(N_x=100, s_bar = 10, label="a") #Figure12.3(a)
    sample50 = figure12_3(N_x=100, s_bar = 50, label="b") #Figure12.3(b)
    figure12_4(N_x=100, s_bar = 10, label="a",sample0=sample10) #Figure12.4(a)
    figure12_4(N_x=100, s_bar = 50, label="b",sample0=sample50) #Figure12.4(b)











