# -----------------------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     16 May 2016
#
# File: lab_44.py
#
# This program runs the MAP classifier over the iris dataset.
#
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import multivariate_normal

IRIS_TYPES = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


class MAP():
    def __init__(self, X, y):
        self.norms = {}
        for iris in IRIS_TYPES.values():
            data = X[np.where(y == iris)]
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            self.norms[iris] = multivariate_normal(
                mean=mean, cov=cov)

    def predict(self, X):
        predictions = []
        for x in X:
            chances = [norm.pdf(x) for norm in self.norms.values()]
            predictions.append(np.argmax(chances))
        return predictions


def convert_iris_string(s):
    if IRIS_TYPES.has_key(s):  # noqa
        return IRIS_TYPES[s]
    else:
        return -1


def main():
    Xy = np.loadtxt('./data/iris.data', delimiter=',', dtype=float,
                    converters={4: convert_iris_string})
    X = Xy[:, :4]
    y = Xy[:, 4:].reshape((Xy.shape[0],))

    # generate a random order for the test and learning set
    idx = np.random.permutation(np.arange(Xy.shape[0]))
    Lidx = idx[:90]
    Tidx = idx[90:]

    # Define the learn and the test sets
    X_learn = X[Lidx]
    y_learn = y[Lidx]
    X_test = X[Tidx]
    y_test = y[Tidx]

    # Fit data.
    clf = MAP(X_learn, y_learn)

    # Test all data.
    predictions = clf.predict(X_test)

    # Create confusion matrix.
    cm = np.zeros((3, 3))
    for prediction, truth in zip(predictions, y_test):
        cm[truth, prediction] += 1

    return cm

if __name__ == '__main__':
    cm = main()
    for _ in range(1000):
        cm += main()

    print(cm)
    print('%d tests, of which %d are correct: %f accuracy' %
          (np.sum(cm), np.trace(cm), np.trace(cm)/np.sum(cm)))
