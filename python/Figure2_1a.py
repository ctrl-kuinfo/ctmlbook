# Author: Kenji Kashima
# Date  : 2025/06/21

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
import sys
sys.path.append("./")
import config

def figure2_1a():
    figsize = config.global_config(type=1)

    x = np.linspace(0, 10, 1000)

    pdf_x2 = norm.pdf(x, 0, np.sqrt(2))   # N(0, 2)

    # Laplace(0, 1) 
    pdf_x3 = laplace.pdf(x, 0, 1)

    plt.figure(figsize=figsize)


    plt.plot(np.abs(x), pdf_x2, label=r'$\mathcal{N}(0, 2)$', color='orange')
    plt.plot(np.abs(x), pdf_x3, label=r'${\rm Lap}(0, 1)$', color='purple')

    plt.legend()
    plt.xlabel(r'$\vert {\rm x}\vert$')
    plt.xlim([0,10])
    plt.ylim([1e-12,1])
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/Figure2_1a.pdf")
    plt.show()

if __name__ == '__main__':
    figure2_1a()
