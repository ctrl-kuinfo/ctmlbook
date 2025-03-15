# Author: Kenji Kashima
# Date  : 2023/11/05
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

np.random.seed(23)
import sys
sys.path.append("./")
import config


def figure10_2b(N_k= 300):
    figsize = config.global_config(type= 1)
    # Figure 10.2 (a) transition probability
    P = np.array([[1/3,    1/3,      0,      0],
                  [0  ,    1/3,    1/3,      0],
                  [0  ,    1/3,    1/3,    1/3],
                  [2/3,      0,    1/3,    2/3]])
    
    p_stable = np.ones((4,1))/4
    for _ in range(100):
        p_stable =  P@p_stable
    print("p_100=",p_stable)
    
    # accumulated transition probability
    m,n =P.shape
    P_accum = np.zeros((m,n))
    P_accum[0,:] = P[0,:]
    for i in range(1,m):
        P_accum[i,:]= P_accum[i-1,:]+P[i,:]


    p_list = np.zeros(N_k) 
    p_list[0] = 4 # start at 4

    for i in range(1,N_k):
        u = np.random.rand()
        T = P_accum[:,int(p_list[i-1])-1] #transition probability
        for j in range(m):
            if u <= T[j]:
                p_list[i] = j+1
                break

    plt.figure(figsize=figsize)
    plt.stairs(p_list,linewidth=2,zorder = 10,baseline=4)
    plt.ylim([0.8,4.2])
    plt.xlim([0,N_k])
    plt.xticks(np.arange(0,N_k+1,10))
    plt.xlabel(r"$k$")
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("./figures/Figure10_2b.pdf")
    plt.show()

if __name__ == '__main__':
    figure10_2b(N_k=50)










