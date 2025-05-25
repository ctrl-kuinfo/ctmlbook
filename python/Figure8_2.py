# Author: Kenji Kashima
# Date  : 2025/05/01

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)
import sys
sys.path.append("./")
import config

def sysid_module(p_star:np.ndarray,n:int,q0:np.ndarray,u:np.ndarray,v:np.ndarray,p0:np.ndarray,Sigma0:np.ndarray,alpha:float):
    """
        System identification
    % args: p_star -- the TRUE system parameters [a1,a2,...,a_n,b1,b2,...,b_m]
    %       n      -- the number of a  (m denotes the number of b)
    %       q0     -- initial data [y_n,...,y_0,u_m,...,u_0]
    %       u      -- u_data
    %       v      -- v_data
    %       p0     -- initial parameters 
    %       Sigma0 -- initial Sigma in Algorithm 3
    %       alpha   -- forgetting factor
    """

    N, dim_p= np.shape(p_star) # dim_p = m + n  ; N denote the length of the data series
    y = np.zeros(N-1) 
    q = np.zeros((N,dim_p))
    q[0,:] = q0

    # open loop control to genereate y data
    for k in range(N-1):
        y[k] =  p_star[k,:] @ q[k,:] + v[k]
        q[k+1,:]=np.hstack([y[k],q[k,0:n-1],u[k+1],q[k,n:dim_p-1]])

    p_est = np.zeros((N,dim_p))
    err_a = np.zeros(N)
    err_b = np.zeros(N)
    TrSigma = np.zeros(N)

    p_est[0,:] = p0 
    Sigma = Sigma0

    # sum of the squares between the true parameters and the estimated parameters
    err_a[0] = np.sum((p_star[0,0:n]-p0[0:n])**2)
    err_b[0] = np.sum((p_star[0,n:dim_p]-p0[n:dim_p])**2)

    TrSigma[0] = np.trace(Sigma)

    for k in range (N-1):
        # Algorithm 3:
        H = Sigma@q[k,:] / (alpha + q[k,:]@Sigma@q[k,:])  #line 3
        p_est[k+1,:] = p_est[k,:]+H*(y[k]-p_est[k,:]@q[k,:]) #line 4
        Sigma = (Sigma - (np.expand_dims(H,0).T @ np.expand_dims(q[k,:],0)).squeeze()@Sigma)/alpha #line 5
 
        # for figure 8.3 (c)~(f)
        err_a[k+1] = np.sum((p_star[k+1,0:n]-p_est[k+1,0:n])**2) 
        err_b[k+1] = np.sum((p_star[k+1,n:dim_p]-p_est[k+1,n:dim_p])**2)
        TrSigma[k+1] = np.trace(Sigma) # trace of Sigma
    return err_a,err_b,TrSigma,p_est,y


def figure8_2a(N_k=1000):
    """
        N_k - number of k
    """
    figsize = config.global_config(type= 0)
    plt.figure(figsize=figsize)
    # the true system y_k = q_k^T * p^*
    # where q_k = [y_k, y_{k-1}, y_{k-2}, u_k, u_{k-1}]
    alpha=1
    a=[1.2,-0.47,0.06] # coefficient of (z-0.5)(z-0.4)(z-0.3)
    b=[1.0,2.0]        # u1,u0
    n, m = len(a), len(b)
    p_star= np.array([a+b]).repeat(N_k,axis=0) # p^* the true parameter
    q0=np.array([0,0,0,1,1]) # initial states and inputs, note that u0=u1=1
    p0=np.array([0,0,0,0,0]) # initial parameters are zeors
    Sigma0 = 1e4*np.eye(m+n)# See Algorithm 3
    
    # u_k = 1, v_k~N(0,1)
    sigma_v = 1
    v = np.random.randn(N_k)*sigma_v # random noise N(0,sigma_v^2)
    u = np.ones(N_k)              # constant input u_k = 1 
    _,_,_,_,y = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(y,label=r'$u_k=1,v_k\sim {\mathcal N}(0,1)$',linewidth=0.5)

    # u_k ~ N(1,1), v_k~N(0,1)
    sigma_v = 0.1
    v = np.random.randn(N_k)*sigma_v # random noise N(0,sigma_v^2)
    u = np.ones(N_k) + np.random.randn(N_k)  # random input N(1,1)
    _,_,_,_,y = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(y,label=r'$u_k \sim {\mathcal N}(1,1), v_k\sim {\mathcal N}(0,0.01)$',linewidth=0.5)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$y_k$")
    plt.xlim([0,N_k])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure8_2a.pdf")
    plt.show()

def figure8_2b(N_k=5000):
    """
        N_k - number of k
    """
    figsize = config.global_config(type= 0)
    plt.figure(figsize=figsize)
    # the true system y_k = q_k^T * p^*
    # where q_k = [y_k, y_{k-1}, y_{k-2}, u_k, u_{k-1}]
    a=[1.2,-0.47,0.06] # coefficient of (z-0.5)(z-0.4)(z-0.3)
    a_prime=[1.0,-0.47,0.06] 
    b=[1.0,2.0]        # u1,u0
    n, m = len(a), len(b)
    p_star = np.array(a+b) # p^* the true parameter
    for k in range(1,N_k):
        if k % 2000 > 1000:
            p_star = np.vstack((p_star,a_prime+b))
        else:
            p_star = np.vstack((p_star,a+b))

    q0=np.array([0,0,0,1,1]) # initial states and inputs, note that u0=u1=1
    p0=np.array([0,0,0,0,0]) # initial parameters are zeors
    Sigma0 = 1e4*np.eye(m+n)# See Algorithm 3
    
    # u_k ~ N(0,1), v_k ~ N(0,1)
    sigma_v = 0.1
    v = np.random.randn(N_k)*sigma_v # random noise N(0,sigma_v^2)
    u = np.random.randn(N_k)         # constant input u_k ~ N(0,1) 

    alpha = 1.0
    _,_,_,p_est,_ = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=1$',linewidth=0.5)

    alpha = 0.995
    _,_,_,p_est,_ = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=0.995$',linewidth=0.5)
  
    alpha = 0.8
    _,_,_,p_est,_ = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.plot(p_est[:,0],label=r'$\alpha=0.8$',linewidth=0.5)
    
    plt.plot(p_star[:,0],label = "True")

    plt.xlabel(r"$k$")
    plt.ylabel(r"${\rm a}_1$")
    plt.xlim([0,N_k])
    plt.ylim([0.8,1.4])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure8_2b.pdf")
    plt.show()

def figure8_2cdef(N_k=100000, label = "c"):
    """
        N_k - number of k
    """
    figsize = config.global_config(type= 0)
    # the true system y_k = q_k^T * p^*
    # where q_k = [y_k, y_{k-1}, y_{k-2}, u_k, u_{k-1}]
    a=[1.2,-0.47,0.06] # coefficient of (z-0.5)(z-0.4)(z-0.3)
    b=[1.0,2.0]        # u1,u0
    n, m = len(a), len(b)

    alpha = 1.0
    p_star= np.array([a+b]).repeat(N_k,axis=0) # p^* the true parameter
    q0=np.array([0,0,0,1,1]) # initial states and inputs, note that u0=u1=1
    p0=np.array([0,0,0,0,0]) # initial parameters are zeors
    Sigma0 = 1e4*np.eye(m+n)# See Algorithm 3
    
    if label == "c" or label == "d":
        sigma_v = 1
    else:
        sigma_v = 0.1

    if label == "c" or label == "e":
        # u_k = 1, v_k~N(0,sigma_v^2)
        v = np.random.randn(N_k)*sigma_v # random noise N(0,sigma_v^2)
        u = np.ones(N_k)         # constant input u_k ~ N(0,1) 
    else:
        # u_k ~ N(1,1), v_k~N(0,sigma_v^2)
        v = np.random.randn(N_k)*sigma_v # random noise N(0,sigma_v^2)
        u = np.ones(N_k) + np.random.randn(N_k)         # constant input u_k ~ N(0,1) 

    plt.figure(figsize=figsize)
    err_a,err_b,TrSigma,_,_ = sysid_module(p_star,n,q0,u,v,p0,Sigma0/sigma_v,alpha)
    plt.loglog(err_a,label=r'$\|\check{\rm p}^a - {\rm a}^*\|^2$',linewidth=1)
    plt.loglog(err_b,label=r'$\|\check{\rm p}^b - {\rm b}^*\|^2$',linewidth=1)
    plt.loglog(TrSigma*sigma_v,label=r'${\rm Trace}(\check\Sigma)$',linewidth=1)


    plt.xlabel(r"$k$")
    plt.xlim([0,N_k])
    plt.ylim([1e-10,1e5])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure8_2{}.pdf".format(label))

    plt.show()


if __name__ == '__main__':
    figure8_2a(N_k=1000)
    figure8_2b(N_k=5000)
    figure8_2cdef(N_k=100000,label="c")
    figure8_2cdef(N_k=100000,label="d")
    figure8_2cdef(N_k=100000,label="e")
    figure8_2cdef(N_k=100000,label="f")
