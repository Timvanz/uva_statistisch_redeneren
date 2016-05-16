# -----------------------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     16 May 2016
#
# File: lab_43.py
#
# In this program P_{XC}(x, C=1), P_{XC}(x, C=2), P(C=1|x) and P(C=2|x) are
# plotted.
#
# -----------------------------------------------------------------------------
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def main():
    # p_1 is the normal distribution for mu=4, sigma=1 and a P(C=1)=0.3.
    # p_2 is the normal distribution for mu=7, sigma=2 and a P(C=2)=0.7.
    x = sp.linspace(-4, 15, 100)
    p_1 = norm(loc=4, scale=1).pdf(x) * 0.3
    p_2 = norm(loc=7, scale=1.5).pdf(x) * 0.7

    # p_1 and p_2 are plotted.
    plt.plot(x, p_1,
             'b-', label='$P_{XC}(x, C=1)$')
    plt.plot(x, p_2,
             'r-', label='$P_{XC}(x, C=2)$')
    # P(C=1|x) and P(C=2|x) plotted.
    # P(C=1|x) = p_1 / (p_1 + p_2)
    # P(C=2|x) = p_2 / (p_1 + p_2)
    plt.plot(x, p_1 / (p_1 + p_2), 'b--', label='$P(C=1|X)$')
    plt.plot(x, p_2 / (p_1 + p_2), 'r--', label='$P(C=2|X)$')
    plt.title("Normal distributions of the example and their class boundaries"
              "\nFor P(C=1)=0.3, $\mu_1=4$ and $\sigma_1=1$"
              " and for P(C=2)=0.7, $\mu_2=7$ and $\sigma_2=2$")
    plt.legend()
    plt.xlabel("x")
    plt.xlim(-4, 15)
    plt.ylim(0, 1)
    plt.savefig("min_err_class.png")

if __name__ == '__main__':
    main()
