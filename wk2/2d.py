# -----------------------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
#
# Date:     7 April 2016
#
# File: 2d.py
#
# Doesn't work yet...
# -----------------------------------------------------------------------------
import numpy as np

def coin_binomial(n, p, times):
	return np.random.binomial(n, p, times)

if __name__=='__main__':
	times = 10000
	n_range = 10
	p_range = 10
	p_foreach_n = []
	for n in range(1, n_range + 1):
		p_for_this_n = []
		for p in range(1, p_range):
			binom = coin_binomial(n, p / float(p_range - 1) , times)
			first = np.sum((n - np.array(binom)) / float(n) / float(times))
			second =  np.sum(binom / float(n) / float(times))
			p_for_this_n.append((first, second))
		p_foreach_n.append(np.array(p_for_this_n) / n)
	""" This 2d array will contain tuples of each change, the change of heads
		or tails """
	print p_foreach_n
	
	for n in p_foreach_n:
		for p in n:
			print p[0] + p[1]
