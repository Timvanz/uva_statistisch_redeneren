import numpy as np
import scipy.stats as spst
import random

SAMPLE_NUM = 100
SAMPLE_SIZE = 50
SAMPLE_PROBABILITY = 0.95


def test_sample(data_mu, sample, interval):
    sample_mu = np.mean(sample)
    sample_std = np.std(sample)
    Z = (sample_mu - data_mu) / (sample_std / np.sqrt(SAMPLE_SIZE))
    return (Z >= interval[0] and Z <= interval[1])


def main():
    data = np.genfromtxt('./tijden-medium.log', dtype=[('time', float)])
    data_mu = np.mean(data['time'])
    interval = spst.t.interval(SAMPLE_PROBABILITY, SAMPLE_SIZE)
    hits = 0
    for _ in range(SAMPLE_NUM):
        if test_sample(data_mu,
                       np.asarray(random.sample(data['time'], SAMPLE_SIZE)),
                       interval):
            hits += 1

    print("Estimate probability: %f" % SAMPLE_PROBABILITY)
    print("Actual probability of %d samples of size %d: %f" %
          (SAMPLE_NUM, SAMPLE_SIZE, (float(hits)/SAMPLE_NUM)))

if __name__ == '__main__':
    main()
