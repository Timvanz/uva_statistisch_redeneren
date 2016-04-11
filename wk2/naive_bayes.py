# -------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
# Date:     11 April 2016
# File: naive_bayes.py
# -------------------------------------------------------------
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

sexes = {'M': {'label': 'Man', 'color': 'b'},
         'F': {'label': 'Vrouw', 'color': 'm'}}

attrs = {'sex': 'Geslacht', 'weight': 'Gewicht (kg)',
         'length': 'Lengte (cm)', 'footsize': 'Schoenmaat'}


def main():
    data14 = np.genfromtxt('./biometrie2016.csv', delimiter=',',
                           skip_header=1, dtype=[
                               ('sex', "|S1"), ('weight', int),
                               ('length', int), ('footsize', 'float')])
    data16 = np.genfromtxt('./biometrie2014.csv', delimiter=',',
                           skip_header=1, dtype=[
                               ('sex', "|S1"), ('weight', int),
                               ('length', int), ('footsize', 'float')])

    ssummary = sex_summary(data14)
    plot(ssummary)
    sex_correct = {'M': 0, 'F': 0}

    for sex in sex_correct.keys():
        for row in data16[np.where(data16['sex'] == sex)]:
            if (row['sex'] == predict(row, ssummary)):
                sex_correct[sex] += 1

    print("      | Man | Vrouw")
    print("Man   | %2d | %2d" % (
        sex_correct['M'],
        (len(data16[np.where(data16['sex'] == 'M')]) - sex_correct['M'])))
    print("Vrouw | %2d | %2d" % (
        (len(data16[np.where(data16['sex'] == 'M')]) - sex_correct['M']),
        sex_correct['F']))


def plot(summary):
    """
    Create subplots for all the attributes of the trainingset
    """
    for sex, val in summary.iteritems():
        for i, (attr, (mean, std)) in enumerate(val.iteritems()):
            plt.subplot(len(val), 1, i+1)
            x = np.linspace(sp.norm(mean, std).ppf(0.001),
                            sp.norm(mean, std).ppf(0.999), 1000)
            plt.plot(x, sp.norm(mean, std).pdf(x), sexes[sex]['color'],
                     label=sexes[sex]['label'])
            plt.fill_between(x, 0, sp.norm(mean, std).pdf(x), alpha=.3,
                             color=sexes[sex]['color'])
            plt.ylim(ymin=0)
            plt.ylabel('Kansdichtheidsfunctie')
            plt.xlabel(attrs[attr])
            plt.legend()
    plt.show()


def predict(x, sex_summary):
    """
    Make a prediction for inputvector x, based on the traindata.
    """
    sex_prob = {}

    for sex, summary in sex_summary.iteritems():
        sex_prob[sex] = 1
        for attr, (mean, std) in summary.iteritems():
            # For all attributes of calculate the probality that
            sex_prob[sex] *= sp.norm(mean, std).pdf(x[attr])

    return max(sex_prob.iteritems(), key=lambda x: x[1])[0]


def sex_summary(data):
    """
    Calculate the mean and standard deviation of both the set of male and the
    set of female.
    """
    ss_data = {
        'M': data[np.where(data['sex'] == 'M')],
        'F': data[np.where(data['sex'] == 'F')]
    }

    summary = {}
    for sex, val in ss_data.iteritems():
        summary[sex] = {
            'weight': (np.mean(val['weight']), np.std(val['weight'])),
            'length': (np.mean(val['length']), np.std(val['length'])),
            'footsize': (np.mean(val['footsize']), np.std(val['footsize'])),
        }
    return summary


if __name__ == '__main__':
    main()
