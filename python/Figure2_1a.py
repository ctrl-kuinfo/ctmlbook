import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
import sys
sys.path.append("./")
import config

def figure2_1a():
    figsize = config.global_config(type=1)

    x = np.linspace(-10, 10, 1000)

    # N(-2, 1), N(2, 1)
    pdf_x1 = norm.pdf(x, -2, 1)  # N(-2, 1)
    pdf_x2 = norm.pdf(x, 2, 1)   # N(2, 1)

    # (x1 + x2) ~ N(0, sqrt(2))
    pdf_x1_plus_x2 = norm.pdf(x, 0, np.sqrt(2))

    # 0.5*N(-2, 1) + 0.5*N(2, 1) 
    pdf_mixture = 0.5 * norm.pdf(x, -2, 1) + 0.5 * norm.pdf(x, 2, 1)

    # Laplace(0, 1) 
    pdf_x3 = laplace.pdf(x, 0, 1)

    plt.figure(figsize=figsize)

    plt.plot(x, pdf_x1, label=r'$x_1 \sim \mathcal{N}(-2, 1)$', color='blue')
    plt.plot(x, pdf_x2, label=r'$x_2 \sim \mathcal{N}(2, 1)$', color='orange')
    plt.plot(x, pdf_x1_plus_x2, label=r'$(x_1+x_2) \sim \mathcal{N}(0, 2)$', color='green')
    plt.plot(x, pdf_mixture, label=r'$(\varphi_{x_1}+\varphi_{x_2})/2$', color='red')
    plt.plot(x, pdf_x3, label=r'$x_3\sim {\rm Lap}(0, 1)$', color='purple')

    plt.legend()
    plt.xlabel(r'${\rm x}$')
    plt.xlim([-5,5])
    plt.ylim([0,1])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/Figure2_1a.pdf")
    plt.show()

if __name__ == '__main__':
    figure2_1a()
