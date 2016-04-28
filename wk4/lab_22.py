
import pylab as pl
#import numpy

if __name__=='__main__':
    mu = pl.array([[2], [8], [16], [32]])
    #print mu
    Sigma = pl.array([[6,2,2,6],[1,4,4,1],[10,1,10,1], [1,4,1,8]])
    d, U = pl.eig(Sigma)
    L = pl.diagflat(d)
    A = pl.dot(U, pl.sqrt(L))
    
    N = []
    mu_deviations = []
    Sigma_deviations = []
    for i in range(1, 40):
        means = pl.array([])
        covariances = pl.array([])
        N.append(50 * i)
        for j in range(1, 200):
            X = pl.randn(4, 50 * i)
            Y = pl.dot(A, X) + pl.tile(mu, 50 * i)
            mean = pl.mean(Y, axis=1)
            covariance = pl.cov(Y)
            covariance = covariance.reshape((1,16))
            #print covariance
            if (len(means) == 0 and len(covariances) == 0):
                means = mean
                covariances = covariance
            else:
                means = pl.vstack((means, mean))
                covariances = pl.vstack((covariances, covariance))
        mu_deviations.append(pl.mean(pl.std(covariances, axis=0)))
        Sigma_deviations.append(pl.mean(pl.std(means, axis=0)))
            #print covariances
    #print N
            #print pl.mean(Y, axis=1)
            #print pl.cov(Y)
    
    
    pl.figure(1)
    pl.clf()
    pl.plot(N, mu_deviations, label="average mean deviation")
    pl.plot(N, Sigma_deviations, label="average covariance deviation")
    pl.legend()
    pl.show()
    