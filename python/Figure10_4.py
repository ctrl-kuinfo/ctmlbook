# Author: Kenji Kashima
# Date  : 2025/05/25
# Note  : You should install scipy first.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator

np.random.seed(13)
import sys
sys.path.append("./")
import config

def L_KL(v, beta, cost, P):
    """ L_KL(v) % find the solution of equation (10.21)"""
    return np.sum((v - cost + np.log(P.T @ np.exp(- beta*v)))**2)
         
def L_IRL(v, beta, a, b, P):
    """ L_IRL(v)  % find the solution of equation below (10.32)"""
    return beta * a @ v + b @ np.log(P.T @ np.exp(-beta * v))

def figure10_4(N_k= 1000, sigma = 1.0):
    
    # Figure 10.2 (a) transition probability(P0)
    # P(i,j) = Prob(state=i to state=j)
    P = np.array([[1/3,    1/3,      0,      0],
                  [0  ,    1/3,    1/3,      0],
                  [0  ,    1/3,    1/3,    1/3],
                  [2/3,      0,    1/3,    2/3]])
    # accumulated transition probability
    # P_accum(i,j) = Prob(state=i to state <= j)
    m,n =P.shape
    beta = 0.8
    invbeta = 1/beta
    sigma = sigma
    cost = np.array([1, 2, 3, 4]) * sigma

    results = minimize( fun = L_KL,
                        x0 = np.ones(4)*5,
                        args = (beta, cost, P),
                        method = "L-BFGS-B",
                        options = {
                            'gtol': 1e-10,    
                            'ftol': 1e-14,    
                            'maxiter': 1000, 
                            'maxls': 50       
                        }
                    )
    print("error=",results.fun)
    print(results.x)
    z_opt = np.exp(-beta*results.x) 
    P_opt = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            P_opt[i,j] = P[i,j] / (P[:,j]@(z_opt/z_opt[i]))
    
    print("sigma:",sigma, "Ppi:\n", P_opt)

    # 固有値と固有ベクトルを計算
    eigvals, eigvecs = np.linalg.eig(P_opt)
    index = np.argmax(eigvals)
    eigenvector_for_1 = eigvecs[:, index]
    sum_of_elements = np.sum(eigenvector_for_1)
    p_stationary = eigenvector_for_1 / sum_of_elements
    print("p_stationary for P_opt：")
    print(p_stationary)

    # p_stable = np.ones((4,1))/4
    # for _ in range(100):
    #     p_stable =  P_opt@p_stable
    # print("p^star_100=",p_stable)

    # accumulated transition probability
    # P_accum(i,j) = Prob(state=i to state <= j)
    P_accum = np.zeros((m,n))
    P_accum[0,:] = P_opt[0,:]
    for i in range(1,m):
        P_accum[i,:] = P_accum[i-1,:] + P_opt[i,:]

    p_list = np.zeros(N_k+1,dtype=np.int8) 
    p_list[0] = 4 
    # start at 4
    a, b=np.array([0,0,0,1]), np.array([0,0,0,0])
    inv_l = np.zeros(n)
    inv_l_hist =np.zeros((4,N_k))


    # Solve L_IRL = 0 with V[1] = 'offset'
    offset = 0

    for i in range(N_k):
        results = minimize(fun=L_IRL,x0= np.array([3,4,10,10]), args=(beta,a,b,P),
                        constraints= {'type': 'eq', 'fun': lambda x: x[0]-offset} )
        results_v = results.x
        inv_z_opt = np.exp( -beta*results_v )  
        inv_l = -np.log(inv_z_opt ** invbeta / (P.T @ inv_z_opt))

        inv_l_hist[:,i] = inv_l

        u = np.random.rand()                # uniform distribution on [0,1]
        T = P_accum[:,int(p_list[i])-1]     # T(j) = Prob( state=i to state <= j )
        for j in range(m):
            if u <= T[j]:
                p_list[i+1] = j+1
                break
        #record how many times of the state has been visited
        a[p_list[i+1]-1] +=1 
        b[p_list[i]-1] +=1

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    plt.stairs(p_list,linewidth=2,zorder = 10,baseline=4)
    plt.ylim([0.8,4.2])
    plt.xlim([0,50])
    plt.xticks(np.arange(0,51,50))
    
    plt.xlabel(r"$k$")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure10_4a.pdf")

    figsize = config.global_config(type=1)
    plt.figure(figsize=figsize)
    
    # state cost for state 1 is fixed to 1
    inv_l_hist = inv_l_hist - inv_l_hist[0,:] + 1
    for i in range(4):
        plt.plot(inv_l_hist[i,:],label=r"$\ell_{}$".format(i+1))
    plt.ylim([0,6.5])
    plt.yticks(np.arange(0,6,1))
    plt.xlim([0,N_k])
    plt.xlabel(r"$k$")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./figures/Figure10_4b.pdf")
    plt.show()

if __name__ == '__main__':
    figure10_4(N_k=1000,sigma=1.0)
