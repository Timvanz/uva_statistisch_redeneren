# -------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     7 April 2016
#
# File: 2d.py
# -------------------------------------------------------------
from scipy.misc import comb


def chance(n, p):
    """
    Calculate the total chance given the number of throws (n)
    and the probability (p).
    """
    total = 0.0
    for k in range(n+1):
        total += comb(n, k, exact=False) * p**k * (1-p) ** (n-k)
    return total


def main():
    for n in range(1, 20):
        for p in [.0, .2, .25, .33, .5, .66, .75, .8, 1.]:
            print('(n,p):(%d,%f) -> %f' % (n, p, chance(n, p)))

if __name__ == '__main__':
    main()
