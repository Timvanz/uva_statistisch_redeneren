# -------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
# Date:     12 April 2016
# File: 21.py
#
# The file for assignment 2.1. Plots N number pairs using
# uniform.
# -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def ibm_rnd(N, seed):
    a = 65539
    c = 0
    m = 2**31
    numbers = [seed]
    for i in range(1, N):
        numbers.append((a * numbers[i - 1] + c) % float(m))
    return np.array(numbers) / m

if __name__=='__main__':
    N = 100
    x = ibm_rnd(N, 983)
    y = ibm_rnd(N, 759)
    plt.title('{0} number pairs created with IBM RND'.format(N))
    plt.plot(x, y, 'o')
    plt.show()