# Author: Kenji Kashima
# Date  : 2023/11/05
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(20)
import sys
sys.path.append("./")
import config


def generate_data_11_4(N_k=4000):
    C = [1.0, 1.0, 1.0, 0.6]
    alpha = [0.3, 1.0, 1.5, 1.0]

    x_list = np.zeros((4,N_k+1))
    y_list = np.zeros((4,N_k))

    x_list[:,0] = np.ones_like(x_list[:,0])*1 #start at 10.0
    for k in range(4):
        for i in range(N_k):        
            x = x_list[k,i]
            y_list[k,i] = x - np.random.randn()   # mean estimation
            x_list[k,i+1] = x - C[k]/((i+1)**alpha[k]) * y_list[k,i]
    return x_list

def figure11_4(x_list,N_k= 100, label="a"):
    figsize = config.global_config(type= 1)
    C = [1.0, 1.0, 1.0, 0.6]
    alpha = [0.3, 1.0, 1.5, 1.0]
    plt.figure(figsize=figsize)  # Please change N_k = 200 to obtain Figure11.4(b)
    for k in range(4):
        plt.plot(x_list[k,:N_k],label=r"$C={}, \alpha={}$".format(C[k],alpha[k]))

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
    x_list = generate_data_11_4()
    figure11_4(x_list, N_k=100,label="a")   
    figure11_4(x_list, N_k=1000,label="b") 
