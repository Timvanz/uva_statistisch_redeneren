# ---------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
# Date:     11 April 2016
# File: 3.py
#
# This program plots the probability distribution function
# and cumulative distribution function from the normal
# distribution. Then takes 1000 values from the normal
# distribution and forms a histogram from those values.
# ---------------------------------------------------------

import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = sp.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    """ Take 1000 values from the normal distribution. """
    r = norm.rvs(size=1000)
    """ Histogram plot. """
    plt.hist(r, normed=True, color='g', label='1000 values')
    """ Plots the probability distribution function. """
    plt.plot(x, norm.pdf(x), 'r-', label='norm pdf')
    """ Plots the cumulative distribution function. """
    plt.plot(x, norm.cdf(x), 'b-', label='norm cdf')
    plt.title('Plot of the normal distibution pdf, cdf and'
              'a histogram of 1000 values')
    plt.legend()
    plt.show()
