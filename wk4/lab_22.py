# -----------------------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     30 april 2016
#
# File: lab_22.py
#
# This file generates random samples as a data matrix from the multivariate
# normal distribution. For different sizes of the data matrix, the average
# covariance deviation and mean deviation is calculated for multiple runs.
# Then, for fixed N, the covariance of the covariance matrix after multiple
# runs is calculated.
# -----------------------------------------------------------------------------

import pylab as pl

if __name__=='__main__':
    mu = pl.array([[2], [8], [16], [32]])
    Sigma = pl.array([[ 3.01602775,  1.02746769, -3.60224613, -2.08792829],
                      [ 1.02746769,  5.65146472, -3.98616664,  0.48723704],
                      [-3.60224613, -3.98616664, 13.04508284, -1.59255406],
                      [-2.08792829,  0.48723704, -1.59255406,  8.28742469]])
    d, U = pl.eig(Sigma)
    L = pl.diagflat(d)
    A = pl.dot(U, pl.sqrt(L))
    
    N = []
    mu_deviations = []
    Sigma_deviations = []
    
    """ First part of the exercise. """
    """ This loop is used to get different sizes of N. """
    for i in range(1, 40):
        means = pl.array([])
        covariances = pl.array([])
        N.append(50 * i)
        """ From this loop, the average is taken to get an accurate measurement. """
        for _ in range(1, 200):
            X = pl.randn(4, 50 * i)
            Y = pl.dot(A, X) + pl.tile(mu, 50 * i)
            mean = pl.mean(Y, axis=1)
            covariance = pl.cov(Y)
            covariance = covariance.reshape((1,16))
            if (len(means) == 0 and len(covariances) == 0):
                means = mean
                covariances = covariance
            else:
                means = pl.vstack((means, mean))
                covariances = pl.vstack((covariances, covariance))
        mu_deviations.append(pl.mean(pl.std(covariances, axis=0)))
        Sigma_deviations.append(pl.mean(pl.std(means, axis=0)))
    
    pl.figure(1)
    pl.clf()
    pl.title('The average deviation, over 200 times,\n of the mean and covariance matrix for a given N')
    pl.xlabel('N')
    pl.ylabel('average deviation')
    pl.plot(N, mu_deviations, label='average mean deviation')
    pl.plot(N, Sigma_deviations, label='average covariance deviation')
    pl.legend()
    pl.savefig('fig22.png')
    
    """ Second part of the exercise. """
    covariances = pl.array([])
    """ Over the loop is iterated to create a data matrix of the covariances
        of the data matrices obtained from the multivariate normal
        distribution. The covariance from this data matrix of covariances is
        shown. """
    for _ in range(1, 200):
        X = pl.randn(4, 1000)
        Y = pl.dot(A, X) + pl.tile(mu, 1000)
        covariance = pl.cov(Y)
        if (len(covariances) == 0):
            covariances = covariance
        else:
            covariances = pl.hstack((covariances, covariance))
    covariance_data = pl.cov(covariances)
    print covariance_data