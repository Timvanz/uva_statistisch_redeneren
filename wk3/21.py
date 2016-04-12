# -------------------------------------------------------------
# Authors:  Tim van Zalingen (10784012)
#           Maico Timmerman (10542590)
# Date:     12 April 2016
# File: 21.py
#
# The file for assignment 2.1. Plots N number pairs using
# uniform.
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
if __name__=='__main__':
	N = 5000
	x = []
	y = []
	for i in range(N):
		x.append(np.random.uniform())
		y.append(np.random.uniform())
	plt.title('{0} number pairs created with numpy.random.uniform()'.format(N))
	plt.plot(x, y, 'o')
	plt.show()