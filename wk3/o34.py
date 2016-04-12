# -------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
# Date:     12 April 2016
# File: 21.py
#
# The file for assignment 3.4. Plots N number pairs using
# exponential distribution.
# -------------------------------------------------------------
from o22 import ibm_rnd
import matplotlib.pyplot as plt
import numpy as np


def exp_rnd(N, seed):
    """
    Calculate N random numbers using seed
    and given lambda.
    """

    def vec_exp(x):
        """
        Inverse transform sampling for the exponential
        distribution
        """
        return -np.log(1-x)

    # Generate N random numbers using normal distribution.
    nums = ibm_rnd(N, seed)
    return np.vectorize(vec_exp)(nums)


if __name__ == '__main__':
    N = 1000
    x = exp_rnd(N, 983)
    y = exp_rnd(N, 759)
    plt.title('{0} number (exponential distributed) pairs created with IBM RND'
              .format(N))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')
    plt.subplot(1, 2, 2)
    plt.hist(x)
    plt.show()
