from pylab import tile, sum, argmin, argmax, argpartition, bincount
class NNb:
    def __init__(self, X, c, k):
        self.n, self.N = X.shape
        self.X = X
        self.c = c
        self.k = k

    def classify(self, x):
        d = self.X - tile(x.reshape(self.n,1), self.N)
        dsq = sum(d*d,0)
        neighbours = self.c[argpartition(dsq, self.k)[:self.k]]
        most_common = argmax(bincount(neighbours.astype(int)))
        return most_common
