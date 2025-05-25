# Author: Kenji Kashima
# Date  : 2025/05/24
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)
import sys
sys.path.append("./")
import config


def figure11_4(N_k= 100,label="a"):
    np.random.seed(23)
    figsize = config.global_config(type= 1)
    C = [0.6, 0.6, 1.0, 1.0]
    alpha = [1.0, 0.3, 1.0, 1.5]

    x_list = np.zeros((4,N_k+1))
    y_list = np.zeros((4,N_k))
    plt.figure(figsize=figsize)  # Please change N_k = 10000 to obtain Figure12.2(b)
    x_list[:,0] = np.ones_like(x_list[:,0])*1 #start at 10.0
    for k in range(4):
        for i in range(N_k):        
            x = x_list[k,i]
            y_list[k,i] = x - np.random.randn()   # mean estimation
            x_list[k,i+1] = x - C[k]/((i+1)**alpha[k]) * y_list[k,i]
        plt.plot(x_list[k,:],label=r"$C={}, \alpha={}$".format(C[k],alpha[k]))

    # plt.scatter(0,10,marker='o',s=80,clip_on=False,color='black',label= "Initial Value",zorder=20)
    plt.ylabel(r"$p_k$")
    plt.xlabel(r"$k$")
    plt.xlim([0,N_k])
    plt.ylim([-2,2])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure11_4{}.pdf".format(label))
    plt.show()

if __name__ == '__main__':
    figure11_4(N_k=100,label="a")   
    figure11_4(N_k=5000,label="b") 
