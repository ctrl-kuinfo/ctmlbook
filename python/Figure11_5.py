# Author: Kenji Kashima
# Date  : 2023/11/05
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)
import sys
sys.path.append("./")
import config


def figure11_5a():
    figsize = config.global_config(type= 1)
    plt.figure(figsize=figsize)
    x = np.arange(-5,5,0.01)
    y = (x**2/2-x*np.cos(x*10)/20+np.sin(x*10)/20)/2
    plt.plot(x,y,label=r'$L(x)$')
    y = (x+x*np.sin(x*10)/2)/2
    plt.plot(x,y,label=r"$\nabla L(x)$")
    y = x + 1/2
    plt.plot(x,y,label=r'$\nabla L_1(x)$')
    y = x*np.sin(x*10)/2 -1/2
    plt.plot(x,y,label=r"$\nabla L_2(x)$")
    
    plt.legend()
    plt.grid()
    plt.xlabel(r"$x$")
    plt.tight_layout()
    plt.savefig("./figures/Figure11_5a.pdf")
    plt.show()

def figure11_5b(N_k= 2000):
    figsize = config.global_config(type= 1)
    C = [1.0, 1.0, 1.0]
    alpha = [0.4,0.8,1.2]

    x_list = np.zeros((3,N_k+1))
    y_list = np.zeros((3,N_k))
    plt.figure(figsize=figsize)  
    x_list[:,0] = np.ones_like(x_list[:,0])*1 #start at 2.0
    for k in range(3):
        for i in range(N_k):        
            x = x_list[k,i]
            randn = np.random.randn()
            if randn< 0:
                y_list[k,i] = x + 1/2 ;   # solving x^2 =2
            else:
                y_list[k,i] = x*np.sin(x*10)/2 - 1/2 ;   # solving x^2 =2

            x_list[k,i+1] = x - C[k]/((i+1)**alpha[k]) * y_list[k,i]
        plt.plot(x_list[k,:],label=r"$\alpha={}$".format(alpha[k]))

    # plt.scatter(0,1,marker='o',s=40,clip_on=False,color='black',label= "Initial Value",zorder=20)
    plt.ylabel(r"$p_k$")
    plt.xlabel(r"$k$")
    plt.xlim([0,N_k])
    plt.ylim([-2,2])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure11_5b.pdf")
    plt.show()


if __name__ == '__main__':
    figure11_5a()
    figure11_5b()