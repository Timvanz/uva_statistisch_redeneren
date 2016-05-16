from sklearn import svm
import numpy as np


def convert_iris_string(s):
    tab = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    if tab.has_key(s):  # noqa
        return tab[s]
    else:
        return -1


def main():
    Xy = np.loadtxt('./iris.data', delimiter=',', dtype=float,
                    converters={4: convert_iris_string})
    X = Xy[:, :3]
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
    clf = svm.SVC()
    clf.fit(X_learn, y_learn)

    # Test all data.
    predictions = clf.predict(X_test)

    # Create confusion matrix.
    cm = np.zeros((3, 3))
    for prediction, truth in zip(predictions, y_test):
        cm[truth, prediction] += 1

    return cm

if __name__ == '__main__':
    # try:
    #     cm = np.load('cm.npy')
    # except:
    cm = main()
    for _ in range(1000):
        cm += main()
        # np.save('cm', cm)

    print('%d tests, of which %d are correct: %f accuracy' %
          (np.sum(cm), np.trace(cm), np.trace(cm)/np.sum(cm)))
    print(cm)
