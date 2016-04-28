
import pylab as pl

if __name__=='__main__':
    mu = pl.array([[2], [8], [16], [32]])
    print mu
    Sigma = pl.array([[6,2,2,6],[1,4,4,1],[10,1,10,1], [1,4,1,8]])
    d, U = pl.eig(Sigma)
    L = pl.diagflat(d)
    A = pl.dot(U, pl.sqrt(L))
    X = pl.randn(4, 1000)
    print X
    Y = pl.dot(A, X) + pl.tile(mu, 1000)
    pl.figure(1)
    pl.clf()
    
    pl.plot(X[0], Y[2], '+', color='#FF0000')
    pl.plot(X[0], Y[3], '+', color='#00FF00')
    pl.plot(X[0], Y[1], '+', color='#0000FF')
    pl.plot(X[1], Y[0], 'x', color='#FFFF00')
    pl.plot(X[1], Y[2], 'x', color='#00FFFF')
    pl.plot(X[1], Y[3], 'x', color='#444444')
    pl.plot(X[2], Y[0], '.', color='#FFFFFF')
    pl.plot(X[2], Y[1], '.', color='#222222')
    pl.plot(X[2], Y[3], '.', color='#AAAAAA')
    pl.plot(X[3], Y[0], '+', color='#FFAA22')
    pl.plot(X[3], Y[1], '+', color='#22AAFF')
    pl.plot(X[3], Y[2], '+', color='#FFDD00')
    
    pl.axis('equal')
    pl.show()
    