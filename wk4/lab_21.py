# -----------------------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     30 april 2016
#
# File: lab_21.py
#
# This file generates random samples as a data matrix from the multivariate
# normal distribution. 12 scatter plots are made for each position in the
# matrix, exctept for the diagonal.
# -----------------------------------------------------------------------------
import pylab as pl


def main():
    mu = pl.array([[0], [12], [24], [36]])
    Sigma = pl.array([[3.01602775,  1.02746769, -3.60224613, -2.08792829],
                      [1.02746769,  5.65146472, -3.98616664,  0.48723704],
                      [-3.60224613, -3.98616664, 13.04508284, -1.59255406],
                      [-2.08792829,  0.48723704, -1.59255406,  8.28742469]])

    # The data matrix is created for above mu and Sigma.
    d, U = pl.eig(Sigma)
    L = pl.diagflat(d)
    A = pl.dot(U, pl.sqrt(L))
    X = pl.randn(4, 1000)

    # Y is the data matrix of random samples.
    Y = pl.dot(A, X) + pl.tile(mu, 1000)

    pl.figure(1)
    pl.clf()
    pl.plot(X[0], Y[1], '+', color='#0000FF', label='i=0,j=1')
    pl.plot(X[0], Y[2], '+', color='#FF0000', label='i=0,j=2')
    pl.plot(X[0], Y[3], '+', color='#00FF00', label='i=0,j=3')
    pl.plot(X[1], Y[0], 'x', color='#FFFF00', label='i=1,j=0')
    pl.plot(X[1], Y[2], 'x', color='#00FFFF', label='i=1,j=2')
    pl.plot(X[1], Y[3], 'x', color='#444444', label='i=1,j=3')
    pl.plot(X[2], Y[0], '.', color='#774411', label='i=2,j=0')
    pl.plot(X[2], Y[1], '.', color='#222222', label='i=2,j=1')
    pl.plot(X[2], Y[3], '.', color='#AAAAAA', label='i=2,j=3')
    pl.plot(X[3], Y[0], '+', color='#FFAA22', label='i=3,j=0')
    pl.plot(X[3], Y[1], '+', color='#22AAFF', label='i=3,j=1')
    pl.plot(X[3], Y[2], '+', color='#FFDD00', label='i=3,j=2')
    pl.legend()
    pl.savefig('fig21.png')

if __name__ == '__main__':
    main()
