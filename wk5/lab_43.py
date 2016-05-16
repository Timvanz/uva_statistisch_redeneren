import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def main():
    x = sp.linspace(-4, 15, 100)
    plt.plot(x, norm(loc=4, scale=1).pdf(x) * 0.3,
             'b-', label='$\mu_1=4, \sigma_1=1$')
    plt.plot(x, norm(loc=7, scale=1.5).pdf(x) * 0.7,
             'r-', label='$\mu_2=7, \sigma_2=1.5$')

    p_1 = norm(loc=4, scale=1).pdf(x) * 0.3
    p_2 = norm(loc=7, scale=1.5).pdf(x) * 0.7

    plt.plot(x, p_1 / (p_1 + p_2), 'b--', label='$\mu_1=4, \sigma_1=1$')
    plt.plot(x, p_2 / (p_1 + p_2), 'r--', label='$\mu_2=7, \sigma_2=1.5$')

    plt.legend()
    plt.xlim(-4, 15)
    plt.ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    main()
