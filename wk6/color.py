#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import svm, grid_search, preprocessing, cross_validation

colors = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink',
          'red', 'violot', 'white', 'yellow']


# does a line of text contains a color name?
def contrains_color(line):
    for c in colors:
        if line.find(c) >= 0:
            return colors.index(c), c
    return None, None


# read the file and store spectra in matrix (rows are the spectra)
# and the classes in vector y
def main():
    with open('./natural400_700_5.asc') as f:
        lines = f.readlines()

        D = np.zeros((0, 61))
        y = np.array([])

        for i in range(0, len(lines), 2):
            ind, c = contrains_color(lines[i])
            if ind is not None:
                d = np.fromstring(lines[i+1], dtype=int, sep=" ")
                D = np.append(D, np.array([d]), axis=0)
                y = np.append(y, ind)

    D = preprocessing.scale(D)
    D_train, D_test, y_train, y_test = cross_validation.train_test_split(
        D, y, test_size=0.2, random_state=42)

    svr = svm.SVC()
    param_grid = dict(gamma=np.logspace(-2, 10, 13), C=np.logspace(-9, 3, 13))
    clf = grid_search.GridSearchCV(svr, param_grid)
    clf.fit(D_train, y_train)

    predictions = clf.predict(D_test)
    cm = np.zeros((len(colors), len(colors)))
    for prediction, truth in zip(predictions, y_test):
        cm[truth, prediction] += 1

    return cm

if __name__ == '__main__':
    cm = main()
    for _ in range(100):
        cm += main()

    print('%d tests, of which %d are correct: %f accuracy' %
          (np.sum(cm), np.trace(cm), np.trace(cm)/np.sum(cm)))
    print(cm)
