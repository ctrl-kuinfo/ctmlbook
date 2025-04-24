# Author: Kenji Kashima
# Date  : 2025/04/01
# Note  : pip install control

import control as ctl
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
import sys
sys.path.append("./")
import config


def figure3_4a():
    figsize = config.global_config(type=1)

    # Chebyshev Type II filter design
    den = [1.0000,-2.2741,1.7904,-0.4801]
    num = [0.0159, 0.0022,0.0022, 0.0159]
    dsys= ctl.tf(num,den,dt=True)

    mag, _, omega = ctl.frequency_response(dsys,np.arange(0,np.pi,0.01))
    plt.figure(figsize=figsize)
    plt.plot(omega,mag,linewidth=1.0,color='blue',label='frequency weight')
    plt.plot([0,0.25,0.6,np.pi],[1,1,0,0],linewidth=2,linestyle='--',color='red',label='prior information')
    plt.xlabel(r'$\varpi$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/Figure3_4a.pdf")
    plt.show()

def figure3_4b(n_k:int=200):
    '''n_k - total steps'''
    figsize = config.global_config(type=1)

    # Chebyshev Type II filter design
    den = [1.0000,-2.2741,1.7904,-0.4801]
    num = [0.0159, 0.0022,0.0022, 0.0159]
    dsys= ctl.tf(num,den,dt=1.0) # discrete-time interval is 1.0s
    dsys = ctl.tf2io(dsys)       # Convert a transfer function into an I/O system 
    v_k = np.random.randn(n_k+1) # random input
    data = ctl.input_output_response(dsys,T=np.arange(0,201),U=u_k)
    y_k = data.y[0,:]
    plt.figure(figsize=figsize)
    plt.xlabel(r'$k$')
    plt.xlim(0,200)
    plt.ylim(-2,2)
    plt.stairs(v_k,label='white')
    plt.stairs(y_k,linewidth=1.0,label='colored')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./figures/Figure3_4b.pdf")
    plt.show()


if __name__ == '__main__':
    figure3_4a()
    figure3_4b(n_k=200)